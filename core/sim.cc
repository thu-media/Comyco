// StreamLens-compatible chunk-level ABR simulator + beam-search expert.
//
// Mirrors the *linear-buffer, no-abandon* path of StreamLens' sabre.py exactly:
//   * network = list of periods (duration_ms, bandwidth_kbps, latency_ms), wrapping around
//   * a request costs one latency (of the period it is issued in) + size / piecewise bandwidth
//   * before every decision the player idles if buffer + segment > buffer_size
//   * the buffer drains while downloading; any shortfall is rebuffering
//   * the first chunk is startup: it is never counted as rebuffering
//
// The per-chunk reward is StreamLens' session QoE
//     QoE = mean_utility - beta * rebuf_ratio - gamma * switch_rate
// rewritten as a sum over chunks (see README):
//     r_i = ln(b_i / b_0) - beta*1000/seg_ms * rebuf_s - gamma*1000/seg_ms * [q_i != q_{i-1}]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

struct Trace {
    std::vector<double> dur, bw, lat;  // ms, kbps (= bits/ms), ms
    double total = 0.0;
};

struct State {
    int period = 0;           // index of current network period
    double time_to_next = 0;  // ms left in the current period
    double buffer_ms = 0;
    int chunk = 0;            // next chunk to download
    int last_q = -1;
};

struct StepOut {
    double download_ms = 0, ttfb_ms = 0, rebuf_ms = 0, wait_ms = 0;
    double size_bits = 0, reward = 0, util = 0;
    bool switched = false, done = false;
};

class Sim {
public:
    Sim(std::vector<std::vector<double>> sizes_bits, double seg_ms, std::vector<double> bitrates_kbps,
        double buffer_size_ms, double beta, double gamma,
        std::vector<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>> traces)
        : sizes_(std::move(sizes_bits)), seg_ms_(seg_ms), bitrates_(std::move(bitrates_kbps)),
          buffer_size_ms_(buffer_size_ms), beta_(beta), gamma_(gamma) {
        if (sizes_.empty()) throw std::invalid_argument("no segments");
        for (const auto& row : sizes_)
            if (row.size() != bitrates_.size()) throw std::invalid_argument("segment row / bitrate mismatch");
        for (const auto& t : traces) {
            Trace tr;
            tr.dur = std::get<0>(t);
            tr.bw = std::get<1>(t);
            tr.lat = std::get<2>(t);
            if (tr.dur.empty() || tr.dur.size() != tr.bw.size() || tr.dur.size() != tr.lat.size())
                throw std::invalid_argument("malformed trace");
            for (double d : tr.dur) tr.total += d;
            traces_.push_back(std::move(tr));
        }
        if (traces_.empty()) throw std::invalid_argument("no traces");
        util_.resize(bitrates_.size());
        for (size_t i = 0; i < bitrates_.size(); ++i) util_[i] = std::log(bitrates_[i]) - std::log(bitrates_[0]);
        w_rebuf_ = beta_ * 1000.0 / seg_ms_;   // per second of stall
        w_switch_ = gamma_ * 1000.0 / seg_ms_;  // per switch
    }

    int n_actions() const { return static_cast<int>(bitrates_.size()); }
    int n_segments() const { return static_cast<int>(sizes_.size()); }
    int n_traces() const { return static_cast<int>(traces_.size()); }

    void reset(int trace_idx, double start_ms) {
        if (trace_idx < 0 || trace_idx >= n_traces()) throw std::out_of_range("trace_idx");
        trace_ = trace_idx;
        const Trace& t = traces_[trace_];
        double off = std::fmod(start_ms, t.total);
        if (off < 0) off += t.total;
        int p = 0;
        while (p + 1 < static_cast<int>(t.dur.size()) && off >= t.dur[p]) {
            off -= t.dur[p];
            ++p;
        }
        s_ = State{};
        s_.period = p;
        s_.time_to_next = t.dur[p] - off;
    }

    // Jump to another position of the video (a seek) with a given buffer. Network state is kept.
    void seek(int chunk, double buffer_ms) {
        if (chunk < 0 || chunk >= n_segments()) throw std::out_of_range("chunk");
        s_.chunk = chunk;
        s_.buffer_ms = buffer_ms;
    }

    StepOut step(int q) { return advance(s_, trace_, q, 1.0, true); }

    std::vector<double> next_sizes() const {
        if (s_.chunk >= n_segments()) return std::vector<double>(bitrates_.size(), 0.0);
        return sizes_[s_.chunk];
    }

    // Beam search over `horizon` chunks. Returns, for each first action, the best cumulative reward found
    // (-1e18 if the beam never kept that action).
    std::vector<double> get_values(int top_k, int horizon, double bw_scale) const {
        const int A = n_actions();
        struct Node {
            State st;
            double total;
            int first;
        };
        std::vector<Node> beam{{s_, 0.0, -1}};
        std::vector<Node> cand;
        const int end = std::min(s_.chunk + horizon, n_segments());
        std::vector<double> best(A, -1e18);
        for (int c = s_.chunk; c < end; ++c) {
            cand.clear();
            cand.reserve(beam.size() * A);
            for (const Node& n : beam) {
                for (int a = 0; a < A; ++a) {
                    State ns = n.st;
                    StepOut o = advance(ns, trace_, a, bw_scale, false);
                    cand.push_back({ns, n.total + o.reward, n.first < 0 ? a : n.first});
                }
            }
            if (static_cast<int>(cand.size()) > top_k) {
                std::nth_element(cand.begin(), cand.begin() + top_k, cand.end(),
                                 [](const Node& x, const Node& y) { return x.total > y.total; });
                cand.resize(top_k);
            }
            beam.swap(cand);
        }
        for (const Node& n : beam)
            if (n.first >= 0) best[n.first] = std::max(best[n.first], n.total);
        return best;
    }

    // Like get_values, but every first action keeps its own beam of top_k / A nodes, so no first action can be
    // crowded out by another one. Nodes tied on reward (to 1e-6) are ranked by buffer. Returns, per first
    // action, the best cumulative reward and the buffer (ms) at the end of that sequence.
    std::pair<std::vector<double>, std::vector<double>> get_values_strat(int top_k, int horizon, double bw_scale) const {
        const int A = n_actions();
        const int per = std::max(1, top_k / A);
        struct Node {
            State st;
            double total;
        };
        auto better = [](const Node& x, const Node& y) {
            const long long qx = std::llround(x.total * 1e6), qy = std::llround(y.total * 1e6);
            if (qx != qy) return qx > qy;
            return x.st.buffer_ms > y.st.buffer_ms;
        };
        const int end = std::min(s_.chunk + horizon, n_segments());
        std::vector<std::vector<Node>> beams(A);
        for (int a = 0; a < A; ++a) {
            State ns = s_;
            StepOut o = advance(ns, trace_, a, bw_scale, false);
            beams[a].push_back({ns, o.reward});
        }
        std::vector<Node> cand;
        for (int c = s_.chunk + 1; c < end; ++c) {
            for (int a = 0; a < A; ++a) {
                cand.clear();
                cand.reserve(beams[a].size() * A);
                for (const Node& n : beams[a]) {
                    for (int b = 0; b < A; ++b) {
                        State ns = n.st;
                        StepOut o = advance(ns, trace_, b, bw_scale, false);
                        cand.push_back({ns, n.total + o.reward});
                    }
                }
                if (static_cast<int>(cand.size()) > per) {
                    std::nth_element(cand.begin(), cand.begin() + per, cand.end(), better);
                    cand.resize(per);
                }
                beams[a].swap(cand);
            }
        }
        std::vector<double> values(A), bufs(A);
        for (int a = 0; a < A; ++a) {
            const Node& best = *std::min_element(beams[a].begin(), beams[a].end(), better);
            values[a] = best.total;
            bufs[a] = best.st.buffer_ms;
        }
        return {values, bufs};
    }

    int chunk() const { return s_.chunk; }
    double buffer_ms() const { return s_.buffer_ms; }
    int last_q() const { return s_.last_q; }
    void set_last_q(int q) { s_.last_q = q; }
    int trace_idx() const { return trace_; }

private:
    void next_period(State& s, const Trace& t) const {
        ++s.period;
        if (s.period == static_cast<int>(t.dur.size())) s.period = 0;
        s.time_to_next = t.dur[s.period];
    }

    // sabre: NetworkModel.do_latency_delay(1)
    double latency_delay(State& s, const Trace& t) const {
        double total = 0, units = 1.0;
        while (units > 0) {
            const double cur = t.lat[s.period];
            const double time = units * cur;
            if (time <= s.time_to_next) {
                total += time;
                s.time_to_next -= time;
                units = 0;
            } else {
                total += s.time_to_next;
                units -= s.time_to_next / cur;
                next_period(s, t);
            }
        }
        return total;
    }

    // sabre: NetworkModel.do_download(size)
    double do_download(State& s, const Trace& t, double size, double scale) const {
        double total = 0;
        while (size > 0) {
            const double bw = t.bw[s.period] * scale;
            if (size <= s.time_to_next * bw) {
                const double time = size / bw;
                total += time;
                s.time_to_next -= time;
                size = 0;
            } else {
                total += s.time_to_next;
                size -= s.time_to_next * bw;
                next_period(s, t);
            }
        }
        return total;
    }

    // sabre: NetworkModel.delay(time)
    void idle(State& s, const Trace& t, double time) const {
        while (time > s.time_to_next) {
            time -= s.time_to_next;
            next_period(s, t);
        }
        s.time_to_next -= time;
    }

    StepOut advance(State& s, int trace_idx, int q, double scale, bool /*real*/) const {
        if (q < 0 || q >= n_actions()) throw std::out_of_range("action");
        if (s.chunk >= n_segments()) throw std::runtime_error("video already finished");
        const Trace& t = traces_[trace_idx];
        StepOut o;
        o.size_bits = sizes_[s.chunk][q];
        o.ttfb_ms = latency_delay(s, t);
        o.download_ms = o.ttfb_ms + do_download(s, t, o.size_bits, scale);

        const bool startup = (s.chunk == 0 && s.last_q < 0 && s.buffer_ms == 0.0);
        if (!startup && o.download_ms > s.buffer_ms) {
            o.rebuf_ms = o.download_ms - s.buffer_ms;
            s.buffer_ms = 0;
        } else if (!startup) {
            s.buffer_ms -= o.download_ms;
        }
        s.buffer_ms += seg_ms_;

        o.util = util_[q];
        o.switched = (s.last_q >= 0 && q != s.last_q);
        o.reward = o.util - w_rebuf_ * (o.rebuf_ms / 1000.0) - (o.switched ? w_switch_ : 0.0);
        s.last_q = q;
        ++s.chunk;
        o.done = (s.chunk >= n_segments());

        // sabre waits *before the next decision* if the buffer would overflow
        if (!o.done) {
            const double full_delay = s.buffer_ms + seg_ms_ - buffer_size_ms_;
            if (full_delay > 0) {
                s.buffer_ms -= full_delay;
                idle(s, t, full_delay);
                o.wait_ms = full_delay;
            }
        }
        return o;
    }

    std::vector<std::vector<double>> sizes_;
    double seg_ms_;
    std::vector<double> bitrates_, util_;
    double buffer_size_ms_, beta_, gamma_, w_rebuf_, w_switch_;
    std::vector<Trace> traces_;
    int trace_ = 0;
    State s_;
};

}  // namespace

PYBIND11_MODULE(slcore, m) {
    m.doc() = "StreamLens-compatible ABR simulator with beam-search expert";
    py::class_<Sim>(m, "Sim")
        .def(py::init<std::vector<std::vector<double>>, double, std::vector<double>, double, double, double,
                      std::vector<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>>(),
             py::arg("sizes_bits"), py::arg("seg_ms"), py::arg("bitrates_kbps"), py::arg("buffer_size_ms"),
             py::arg("beta") = 10.0, py::arg("gamma") = 1.0, py::arg("traces"))
        .def("reset", &Sim::reset, py::arg("trace_idx"), py::arg("start_ms") = 0.0)
        .def("seek", &Sim::seek, py::arg("chunk"), py::arg("buffer_ms"))
        .def("step",
             [](Sim& s, int q) {
                 StepOut o = s.step(q);
                 return py::make_tuple(o.download_ms, o.ttfb_ms, o.rebuf_ms, o.wait_ms, o.size_bits, o.reward,
                                       o.util, o.switched, o.done);
             },
             "returns (download_ms, ttfb_ms, rebuf_ms, wait_ms, size_bits, reward, util, switched, done)")
        .def("next_sizes", &Sim::next_sizes)
        .def("get_values", &Sim::get_values, py::arg("top_k") = 200, py::arg("horizon") = 15,
             py::arg("bw_scale") = 0.9)
        .def("get_values_strat",
             [](const Sim& s, int top_k, int horizon, double bw_scale) {
                 auto r = s.get_values_strat(top_k, horizon, bw_scale);
                 return py::make_tuple(r.first, r.second);
             },
             py::arg("top_k") = 3000, py::arg("horizon") = 15, py::arg("bw_scale") = 0.9,
             "returns (best reward per first action, end buffer in ms of that sequence)")
        .def_property_readonly("chunk", &Sim::chunk)
        .def_property_readonly("buffer_ms", &Sim::buffer_ms)
        .def_property("last_q", &Sim::last_q, &Sim::set_last_q)
        .def_property_readonly("trace_idx", &Sim::trace_idx)
        .def_property_readonly("n_actions", &Sim::n_actions)
        .def_property_readonly("n_segments", &Sim::n_segments)
        .def_property_readonly("n_traces", &Sim::n_traces);
}
