#include <string>
#include <sstream>
#include <vector>
#include <iterator>
using namespace std;
string COOKED_TRACE_FOLDER = './cooked_traces/';

template<typename Out>
void split(const std::string &s, char delim, Out result) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        *(result++) = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

/*function... might want it in some class?*/
int getdir(string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
        files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}

void load_trace(string cooked_trace_folder=COOKED_TRACE_FOLDER, 
    vector<int> &all_cooked_time, vector<int> &all_cooked_bw, vector<int> &all_file_names)
{
    vector<string> files;
    getdir(cooked_trace_folder, files);
    for(auto &cooked_file: files)
    {
        auto file_path = cooked_trace_folder + cooked_file;
        vector<int> cooked_time;
        vector<int> cooked_bw;
        ifstream fin(file_path);
        string line;
        while (getline(fin, line))
        {
            auto parse = split(line, "\t");//line.split()
            cooked_time.push_back(stof(parse[0]))
            cooked_bw.push_back(stof(parse[1]))
        }
        all_cooked_time.push_back(cooked_time)
        all_cooked_bw.push_back(cooked_bw)
        all_file_names.push_back(cooked_file)
}