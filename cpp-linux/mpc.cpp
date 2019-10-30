#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <iterator>
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <deque>
#include "env.hh"
using namespace std;

#define S_INFO 5 //bit_rate, buffer_size, rebuffering_time, bandwidth_measurement, chunk_til_video_end
#define S_LEN 8  //take how many frames in the past
#define A_DIM 6
#define MPC_FUTURE_CHUNK_COUNT 6
//#define ACTOR_LR_RATE 0.0001
//#define CRITIC_LR_RATE 0.001
std::vector<int> VIDEO_BIT_RATE = {300, 750, 1200, 1850, 2850, 4300}; //Kbps
#define BUFFER_NORM_FACTOR 10.0
#define M_IN_K 1000.0
#define RANDOM_SEED 42
#define BITRATE_LEVELS 6
#define DEFAULT_QUALITY 1

//fixed video size -> envivo
std::string VMAF = "./envivo/vmaf/video";
std::string FILESIZE = "./envivo/size/video_size_";
#define CHUNK_TIL_VIDEO_END_CAP 48.0
#define TOTAL_VIDEO_CHUNKS 48

std::string SUMMARY_DIR = "./qoe_vmaf";
std::string LOG_FILE = "./test_results/log_sim_cppmpc";
std::string COOKED_TRACE_FOLDER = "./cooked_traces/";
typedef numeric_limits<double> dbl;
template <typename Out>
void split(const std::string &s, char delim, Out result)
{
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim)
{
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

/*function... might want it in some class?*/
int getdir(string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir.c_str())) == NULL)
    {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        if (dirp->d_name[0] == '.')
            continue; // read . or ..
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

void load_trace(string cooked_trace_folder,
                vector<vector<double>> &all_cooked_time, vector<vector<double>> &all_cooked_bw, vector<string> &all_file_names)
{
    vector<string> files;
    getdir(cooked_trace_folder, files);
    for (auto &cooked_file : files)
    {
        auto file_path = cooked_trace_folder + cooked_file;
        vector<double> cooked_time;
        vector<double> cooked_bw;
        ifstream fin(file_path);
        string line;
        while (getline(fin, line))
        {
            auto parse = split(line, '\t'); //line.split()
            cooked_time.push_back(stod(parse[0]));
            cooked_bw.push_back(stod(parse[1]));
        }
        fin.close();
        all_cooked_time.push_back(cooked_time);
        all_cooked_bw.push_back(cooked_bw);
        all_file_names.push_back(cooked_file);
    }
}

//pass
void readVmaf(unordered_map<int, vector<double>> &vmaf_size)
{
    for (auto bitrate = 0; bitrate < BITRATE_LEVELS; bitrate++)
    {
        vector<double> tmp;
        vmaf_size[bitrate] = tmp;
        ifstream fin(VMAF + to_string(BITRATE_LEVELS - bitrate));
        string s;
        while (getline(fin, s))
        {
            vmaf_size[bitrate].push_back(stod(s));
        }
        fin.close();
    }
}

void readChunk(unordered_map<int, vector<int>> &chunk_size)
{
    for (auto bitrate = 0; bitrate < BITRATE_LEVELS; bitrate++)
    {
        vector<int> tmp;
        chunk_size[bitrate] = tmp;
        ifstream fin(FILESIZE + to_string(bitrate));
        string s;
        while (getline(fin, s))
        {
            chunk_size[bitrate].push_back(stoi(s));
        }
        fin.close();
    }
}

int get_chunk_size(unordered_map<int, vector<int>> &chunk_size, int quality, int index)
{
    if (index < 0 || index > TOTAL_VIDEO_CHUNKS)
        return 0;
    //cout << quality << " " << index << endl;
    return chunk_size[quality][index];
}

double get_chunk_vmaf(unordered_map<int, vector<double>> &vmaf_size, int quality, int index)
{
    if (index < 0 || index > TOTAL_VIDEO_CHUNKS)
        return 0;
    return vmaf_size[quality][index];
}

int main()
{
    unordered_map<int, vector<double>> vmaf_size;
    unordered_map<int, vector<int>> chunk_size;
    readVmaf(vmaf_size);
    readChunk(chunk_size);
    vector<vector<double>> all_cooked_time, all_cooked_bw;
    vector<string> all_file_names;
    load_trace("./norway/", all_cooked_time, all_cooked_bw, all_file_names);
    auto net_env = new Environment(all_cooked_time, all_cooked_bw);
    auto log_path = LOG_FILE + '_' + all_file_names[net_env->trace_idx];
    //log_file = open(log_path, 'w')
    ofstream log_file(log_path);
    double time_stamp = 0.0;

    auto bit_rate = DEFAULT_QUALITY;
    double last_video_vmaf = -1.0;

    auto video_count = 0;
    vector<vector<int>> CHUNK_COMBO_OPTIONS;

    for (auto idx = 0; idx < std::pow(A_DIM, MPC_FUTURE_CHUNK_COUNT); idx++)
    {
        vector<int> vec;
        int j = idx;
        for (auto i = 0; i < MPC_FUTURE_CHUNK_COUNT; ++i)
        {
            auto tmp = j % A_DIM;
            vec.push_back(tmp);
            j /= A_DIM;
        }
        CHUNK_COMBO_OPTIONS.push_back(vec);
    }
    deque<double> past_errors;
    deque<double> past_bandwidth;
    double past_bandwidth_ests = -1;
    while (true)
    {

        double delay;
        double sleep_time;
        double buffer_size;
        double rebuf;
        double video_chunk_size;
        bool end_of_video;
        double video_chunk_remain;
        double video_chunk_vmaf;
        net_env->get_video_chunk(bit_rate, delay, sleep_time, buffer_size, rebuf, video_chunk_size, end_of_video, video_chunk_remain, video_chunk_vmaf);
        //cout << delay << " " << sleep_time << " " << buffer_size << " " << rebuf << " " << video_chunk_size << " " << end_of_video << " " << video_chunk_remain << " " << video_chunk_vmaf << endl;
        time_stamp += delay;      // in ms
        time_stamp += sleep_time; // in ms

        if (last_video_vmaf < 0.0)
            last_video_vmaf = video_chunk_vmaf;

        auto reward = 0.8469011 * video_chunk_vmaf - 28.79591348 * rebuf + 0.29797156 * std::abs(std::max(video_chunk_vmaf - last_video_vmaf, 0.0)) - 1.06099887 * std::abs(std::min(video_chunk_vmaf - last_video_vmaf, 0.0)) -
                      2.661618558192494;
        //cout << reward << endl;
        last_video_vmaf = video_chunk_vmaf;
        log_file.setf(ios::fixed, ios::floatfield);
        //log_file.precision(dbl::digits10);
        log_file << time_stamp / M_IN_K << "\t" << VIDEO_BIT_RATE[bit_rate] << "\t" << buffer_size << "\t" << rebuf << "\t" << video_chunk_size << "\t" << delay << "\t" << reward << "\n";
        log_file.flush();
        //cout << "write" << endl;
        auto bandwidth = video_chunk_size / delay / M_IN_K;
        auto curr_error = 0.0;
        if (past_bandwidth_ests >= 0.0)
            curr_error = std::abs(past_bandwidth_ests - bandwidth) / bandwidth;
        past_errors.push_back(curr_error);
        past_bandwidth.push_back(bandwidth);

        if (past_errors.size() > 3)
            past_errors.pop_front();
        if (past_bandwidth.size() > 5)
            past_bandwidth.pop_front();

        // harmonic mean
        auto bandwidth_sum = 0.0;
        for (auto &p : past_bandwidth)
            bandwidth_sum += (1.0 / p);
        auto harmonic_bandwidth = 1.0 / (bandwidth_sum / past_bandwidth.size());

        // past err
        auto max_error = 0.0;
        for (auto &p : past_errors)
            max_error = std::max(max_error, p);
        auto future_bandwidth = harmonic_bandwidth / (1.0 + max_error); // robustMPC here
        past_bandwidth_ests = future_bandwidth;

        auto last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain - 1);
        auto future_chunk_length = MPC_FUTURE_CHUNK_COUNT;
        if (TOTAL_VIDEO_CHUNKS - last_index - 1 < MPC_FUTURE_CHUNK_COUNT)
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index - 1;

        auto max_reward = -100000000;
        int send_data = 0;
        //best_combo = ()
        auto start_buffer = buffer_size;
        for (auto &combo : CHUNK_COMBO_OPTIONS)
        {

            //cout << "start combo" << endl;
            //auto combo = full_combo;
            double curr_buffer = start_buffer;
            //net_env->reset_download_time();

            //auto curr_rebuffer_time = 0.0;
            //curr_buffer = start_buffer
            double curr_rebuffer_time = 0.0;
            double vmaf_sum = 0.0;
            double vmaf_smoothness0 = 0.0;
            double vmaf_smoothness1 = 0.0;
            double vmaf_last = last_video_vmaf;
            double reward_ = 0.0;
            //for position in range(0, len(combo)):
            for (auto position = 0; position < future_chunk_length; position++)
            {
                auto chunk_quality = combo[position];
                auto index = last_index + position + 1;
                auto sizes = get_chunk_size(chunk_size, chunk_quality, index);
                auto download_time = sizes / future_bandwidth / 1000000.0;
                download_time += 80.0 / 1000.0;
                //double curr_buffer = 0.0;
                if (curr_buffer < download_time)
                {
                    curr_rebuffer_time += (download_time - curr_buffer);
                    curr_buffer = 0.0;
                }
                else
                {
                    curr_buffer -= download_time;
                }
                curr_buffer += 4.0;
                double vmaf_current = get_chunk_vmaf(vmaf_size, chunk_quality, index);
                vmaf_sum += vmaf_current;
                vmaf_smoothness0 += std::abs(std::max(vmaf_current - vmaf_last, 0.0));
                vmaf_smoothness1 += std::abs(std::min(vmaf_current - vmaf_last, 0.0));
                vmaf_last = vmaf_current;
                //early stop
                auto remaining = future_chunk_length - position - 1;
                reward_ = 0.8469011 * vmaf_sum - 28.79591348 * curr_rebuffer_time + 0.29797156 * vmaf_smoothness0 - 1.06099887 * vmaf_smoothness1 -
                          2.661618558192494;
                auto reward_est = reward_ + remaining * 100.0;
                if (reward_est < max_reward)
                    break;
            }
            //auto reward = 0.8469011 * vmaf_sum - 28.79591348 * curr_rebuffer_time + 0.29797156 * vmaf_smoothness0 - 1.06099887 * vmaf_smoothness1 -
            //              2.661618558192494;
            //cout << vmaf_sum << " " << curr_rebuffer_time << " " << reward << endl;
            if (reward_ >= max_reward)
            {
                max_reward = reward_;
                send_data = combo[0];
            }
        }
        bit_rate = send_data;

        if (end_of_video)
        {
            //log_file.write('\n');
            log_file << "\n"
                     << endl;
            log_file.close();

            bit_rate = DEFAULT_QUALITY;
            last_video_vmaf = -1.0;
            cout << "video count " << video_count << endl;
            video_count++;

            if (video_count >= all_file_names.size())
                break;

            log_path = LOG_FILE + '_' + all_file_names[net_env->trace_idx];
            log_file.open(log_path);
        }
    }
    return 0;
}