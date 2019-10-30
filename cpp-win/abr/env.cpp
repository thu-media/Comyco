#include "env.hh"
#include <fstream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <algorithm>
#define MILLISECONDS_IN_SECOND  1000.0
#define B_IN_MB  1000000.0
#define BITS_IN_BYTE  8.0
#define RANDOM_SEED  42
#define VIDEO_CHUNCK_LEN  4000.0  // millisec, every time add this amount to buffer
#define BITRATE_LEVELS  6
#define MPC_FUTURE_CHUNK_COUNT 8 //define future count
#define A_DIM BITRATE_LEVEL
#define BUFFER_THRESH  60.0 * MILLISECONDS_IN_SECOND  // millisec, max buffer limit
#define DRAIN_BUFFER_SLEEP_TIME  500.0  // millisec
#define PACKET_PAYLOAD_PORTION  0.95
#define LINK_RTT  80  // millisec
#define PACKET_SIZE  1500  // bytes

//fixed video size -> envivo
#define VIDEO_SIZE_FILE  "./envivo/size/video_size_"
#define VMAF  "./envivo/vmaf/video"
#define CHUNK_TIL_VIDEO_END_CAP 48.0
#define TOTAL_VIDEO_CHUNKS CHUNK_TIL_VIDEO_END_CAP

#undef max
#undef min

void Environment::split(const std::string &s, char delim, std::vector<std::string>& result)
{
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim))
	{
		result.push_back(item);
	}
}

std::vector<std::string> Environment::split(const std::string &s, char delim)
{
	std::vector<std::string> elems;
	this->split(s, delim, elems);
	return elems;
}

/*function... might want it in some class?*/
int Environment::getdir(string dir, vector<string> &files)
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
		if (dirp->d_name[0] == '.') continue;  // read . or ..
		files.push_back(string(dirp->d_name));
	}
	closedir(dp);
	return 0;
}

Environment::Environment(string filedir)
{
	srand(time(NULL));
	vector<vector<double>> all_cooked_time, all_cooked_bw;
	vector<string> all_file_names;
	this->load_trace(filedir, all_cooked_time, all_cooked_bw, all_file_names);
	this->all_file_names = all_file_names;
	this->init(all_cooked_time, all_cooked_bw);
}

void Environment::load_trace(string cooked_trace_folder,
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

void Environment::init(vector<vector<double>> &all_cooked_time, vector<vector<double>> &all_cooked_bw)
{
	this->all_cooked_time = all_cooked_time;
	this->all_cooked_bw = all_cooked_bw;

	this->video_chunk_counter = 0;
	this->buffer_size = 0;

	// pick a random trace file

	this->trace_idx = rand() % all_cooked_time.size();
	this->cooked_time = this->all_cooked_time[this->trace_idx];
	this->cooked_bw = this->all_cooked_bw[this->trace_idx];

	this->mahimahi_ptr = rand() % (this->cooked_bw.size() - 1) + 1;
	this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr - 1];
	readChunk(this->chunk_size);
	readVmaf(this->vmaf_size);
	this->virtual_mahimahi_ptr = this->mahimahi_ptr;
	this->virtual_last_mahimahi_time = this->last_mahimahi_time;
	//vector<vector<int>> CHUNK_COMBO_OPTIONS;

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
        this->CHUNK_COMBO_OPTIONS.push_back(vec);
    }
}

void Environment::reset_download_time()
{
	this->virtual_mahimahi_ptr = this->mahimahi_ptr;
	this->virtual_last_mahimahi_time = this->last_mahimahi_time;
}

double Environment::get_download_time(int video_chunk_size)
{
	auto delay = 0.0;  // in ms
	auto video_chunk_counter_sent = 0;  // in bytes

	while (true)  // download video chunk over mahimahi
	{
		auto throughput = this->cooked_bw[this->virtual_mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE;
		auto duration = this->cooked_time[this->virtual_mahimahi_ptr] - this->virtual_last_mahimahi_time;

		auto packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION;

		if (video_chunk_counter_sent + packet_payload > video_chunk_size)
		{
			auto fractional_time = (video_chunk_size - video_chunk_counter_sent) / throughput / PACKET_PAYLOAD_PORTION;
			delay += fractional_time;
			this->virtual_last_mahimahi_time += fractional_time;
			break;
		}

		video_chunk_counter_sent += packet_payload;
		delay += duration;
		this->virtual_last_mahimahi_time = this->cooked_time[this->virtual_mahimahi_ptr];
		this->virtual_mahimahi_ptr += 1;

		if (this->virtual_mahimahi_ptr >= this->cooked_bw.size())
		{
			// loop back in the beginning
			// note: trace file starts with time 0
			this->virtual_mahimahi_ptr = 1;
			this->virtual_last_mahimahi_time = 0;
		}
	}
	delay += LINK_RTT / 1000.0;
	return delay;
}


void Environment::get_optimal_v2(double last_video_vmaf, double alpha, double beta, double gamma, double delta)
{
	//auto last_video_vmaf = this->video_chunk_vmaf0;
	auto video_chunk_remain = TOTAL_VIDEO_CHUNKS - this->video_chunk_counter;
	auto last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain - 1);
	auto future_chunk_length = MPC_FUTURE_CHUNK_COUNT;
	if (TOTAL_VIDEO_CHUNKS - last_index - 1 < MPC_FUTURE_CHUNK_COUNT)
		future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index - 1;

	auto max_reward = -100000000;
	int send_data = 0;
	//best_combo = ()
	auto start_buffer = this->buffer_size / MILLISECONDS_IN_SECOND;

	for (auto &combo : this->CHUNK_COMBO_OPTIONS)
	{
		double curr_buffer = start_buffer;
		this->reset_download_time();

		double curr_rebuffer_time = 0.0;
		double vmaf_sum = 0.0;
		double vmaf_smoothness0 = 0.0;
		double vmaf_smoothness1 = 0.0;
		double vmaf_last = last_video_vmaf;
		double reward_ = 0.0;

		for (auto position = 0; position < future_chunk_length; position++)
		{
			auto chunk_quality = combo[position];
			auto index = last_index + position + 1;
			auto sizes = this->chunk_size[chunk_quality][index];
			auto download_time = this->get_download_time(sizes);
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
			double vmaf_current = this->vmaf_size[chunk_quality][index];
			vmaf_sum += vmaf_current;
			vmaf_smoothness0 += std::abs(std::max(vmaf_current - vmaf_last, 0.0));
			vmaf_smoothness1 += std::abs(std::min(vmaf_current - vmaf_last, 0.0));
			vmaf_last = vmaf_current;
			/*
			//early stop
			auto remaining = future_chunk_length - position - 1;
			reward_ = alpha * vmaf_sum + beta * curr_rebuffer_time + gamma * vmaf_smoothness0 + delta * vmaf_smoothness1;
				//-2.661618558192494;
			auto reward_est = reward_ + remaining * 100.0;
			if (reward_est < max_reward)
				break;
			*/
		}
		if (reward_ >= max_reward)
		{
			max_reward = reward_;
			send_data = combo[0];
		}
	}
	this->optimal = send_data;
}
void Environment::get_optimal(double last_video_vmaf)
{
	//auto last_video_vmaf = this->video_chunk_vmaf0;
	auto video_chunk_remain = TOTAL_VIDEO_CHUNKS - this->video_chunk_counter;
	auto last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain - 1);
	auto future_chunk_length = MPC_FUTURE_CHUNK_COUNT;
	if (TOTAL_VIDEO_CHUNKS - last_index - 1 < MPC_FUTURE_CHUNK_COUNT)
		future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index - 1;

	auto max_reward = -100000000;
	int send_data = 0;
	//best_combo = ()
	auto start_buffer = this->buffer_size / MILLISECONDS_IN_SECOND;
	for (auto &combo : this->CHUNK_COMBO_OPTIONS)
	{

		//cout << "start combo" << endl;
		//auto combo = full_combo;
		double curr_buffer = start_buffer;
		this->reset_download_time();

		//auto curr_rebuffer_time = 0.0;
		//curr_buffer = start_buffer
		double curr_rebuffer_time = 0.0;
		double vmaf_sum = 0.0;
		double vmaf_smoothness0 = 0.0;
		double vmaf_smoothness1 = 0.0;
		double vmaf_last = last_video_vmaf;
		double reward_ = 0.0;

		for (auto position = 0; position < future_chunk_length; position++)
		{
			auto chunk_quality = combo[position];
			auto index = last_index + position + 1;
			auto sizes = this->chunk_size[chunk_quality][index];
			auto download_time = this->get_download_time(sizes);
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
			double vmaf_current = this->vmaf_size[chunk_quality][index];
			vmaf_sum += vmaf_current;
			vmaf_smoothness0 += std::abs(std::max(vmaf_current - vmaf_last, 0.0));
			vmaf_smoothness1 += std::abs(std::min(vmaf_current - vmaf_last, 0.0));
			vmaf_last = vmaf_current;
			//early stop
			auto remaining = future_chunk_length - position - 1;
			reward_ = 0.8469011 * vmaf_sum - 28.79591348 * curr_rebuffer_time + 0.29797156 * vmaf_smoothness0 - 1.06099887 * vmaf_smoothness1 -
						2.661618558192494;
			auto reward_est = reward_ + remaining * 100.0;
			if(reward_est < max_reward)
				break;
		}
		if (reward_ >= max_reward)
		{
			max_reward = reward_;
			send_data = combo[0];
		}
	}
	this->optimal = send_data;
}

void Environment::get_video_chunk(int quality)
{
	auto video_chunk_size = this->chunk_size[quality][this->video_chunk_counter];
	auto video_chunk_vmaf = this->vmaf_size[quality][this->video_chunk_counter];

	// use the delivery opportunity in mahimahi
	auto delay = 0.0;  // in ms
	auto video_chunk_counter_sent = 0;  // in bytes

	while (true)  // download video chunk over mahimahi
	{
		auto throughput = this->cooked_bw[this->mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE;
		auto duration = this->cooked_time[this->mahimahi_ptr] - this->last_mahimahi_time;

		auto packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION;

		if (video_chunk_counter_sent + packet_payload > video_chunk_size)
		{
			auto fractional_time = (video_chunk_size - video_chunk_counter_sent) / throughput / PACKET_PAYLOAD_PORTION;
			delay += fractional_time;
			this->last_mahimahi_time += fractional_time;
			break;
		}
		video_chunk_counter_sent += packet_payload;
		delay += duration;
		this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr];
		this->mahimahi_ptr += 1;

		if (this->mahimahi_ptr >= this->cooked_bw.size())
		{
			// loop back in the beginning
			// note: trace file starts with time 0
			this->mahimahi_ptr = 1;
			this->last_mahimahi_time = 0;
		}
	}
	delay *= MILLISECONDS_IN_SECOND;
	delay += LINK_RTT;

	// rebuffer time
	auto rebuf = std::max(delay - this->buffer_size, 0.0);

	// update the buffer
	this->buffer_size = std::max(this->buffer_size - delay, 0.0);

	// add in the new chunk
	this->buffer_size += VIDEO_CHUNCK_LEN;

	// sleep if buffer gets too large
	auto sleep_time = 0.0;
	if (this->buffer_size > BUFFER_THRESH)
	{
		// exceed the buffer limit
		// we need to skip some network bandwidth here
		// but do not add up the delay
		auto drain_buffer_time = this->buffer_size - BUFFER_THRESH;
		sleep_time = std::ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * DRAIN_BUFFER_SLEEP_TIME;
		this->buffer_size -= sleep_time;
		while (true)
		{
			auto duration = this->cooked_time[this->mahimahi_ptr] - this->last_mahimahi_time;
			if (duration > sleep_time / MILLISECONDS_IN_SECOND)
			{
				this->last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND;
				break;
			}
			sleep_time -= duration * MILLISECONDS_IN_SECOND;
			this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr];
			this->mahimahi_ptr += 1;

			if (this->mahimahi_ptr >= this->cooked_bw.size())
			{
				// loop back in the beginning
				// note: trace file starts with time 0
				this->mahimahi_ptr = 1;
				this->last_mahimahi_time = 0;
			}
		}
	}
	// the "last buffer size" return to the controller
	// Note: in old version of dash the lowest buffer is 0.
	// In the new version the buffer always have at least
	// one chunk of video
	auto return_buffer_size = this->buffer_size;

	this->video_chunk_counter += 1;
	auto video_chunk_remain = TOTAL_VIDEO_CHUNKS - this->video_chunk_counter;

	auto end_of_video = false;
	if (this->video_chunk_counter >= TOTAL_VIDEO_CHUNKS)
	{
		end_of_video = true;
		this->buffer_size = 0;
		this->video_chunk_counter = 0;

		this->trace_idx += 1;
		//if(this->trace_idx >= this->all_cooked_time.size())
		//    this->trace_idx = 0;

		this->trace_idx = rand() % all_cooked_time.size();
		this->cooked_time = this->all_cooked_time[this->trace_idx];
		this->cooked_bw = this->all_cooked_bw[this->trace_idx];

		// randomize the start point of the video
		// note: trace file starts with time 0
		//self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
		this->mahimahi_ptr = rand() % (this->cooked_bw.size() - 1) + 1;
		//this->mahimahi_start_ptr;
		this->last_mahimahi_time = this->cooked_time[this->mahimahi_ptr - 1];
	}
	this->delay0 = delay;
	this->sleep_time0 = sleep_time;
	this->return_buffer_size0 = return_buffer_size / MILLISECONDS_IN_SECOND;
	this->rebuf0 = rebuf / MILLISECONDS_IN_SECOND;
	this->video_chunk_size0 = video_chunk_size;
	this->end_of_video0 = end_of_video;
	this->video_chunk_remain0 = video_chunk_remain;
	this->video_chunk_vmaf0 = video_chunk_vmaf;

}

void Environment::readVmaf(unordered_map<int, vector<double>> &vmaf_size)
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

void Environment::readChunk(unordered_map<int, vector<int>> &chunk_size)
{
	for (auto bitrate = 0; bitrate < BITRATE_LEVELS; bitrate++)
	{
		vector<int> tmp;
		chunk_size[bitrate] = tmp;
		ifstream fin(VIDEO_SIZE_FILE + to_string(bitrate));
		string s;
		while (getline(fin, s))
		{
			chunk_size[bitrate].push_back(stoi(s));
		}
		fin.close();
	}
}
