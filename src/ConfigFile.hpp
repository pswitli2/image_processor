#ifndef CONFIGFILE_HPP_
#define CONFIGFILE_HPP_

#include <algorithm>
#include <map>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "Types.hpp"

class ConfigFile
{
public:

    static bool initialize(int argc, const char** argv)
    {
        if (!set_filepath(argc, argv))
        {
            return false;
        }

        if (!parse_file())
        {
            return false;
        }

        return extract_params();
    }

    static const std::filesystem::path input_dir() { return m_input_dir; }
    static const std::filesystem::path output_dir() { return m_output_dir; }
    static DisplayType display_type() { return m_display_type; }
    static double delay() { return m_delay; }
    static const std::string error() { return m_error; }

    static int get_int_param(const std::string& key, int& val)
    {
        if (m_params.find(key) == m_params.end())
        {
            m_error = key + " param error, param doesnt exist";
            return false;
        }

        try
        {
            val = std::stoi(m_params[key]);
        }
        catch(const std::exception& e)
        {
            m_error = key + " param error, unable to convert to int: " + m_params[key];
            return false;
        }

        return true;
    }

    static bool get_double_param(const std::string& key, double& val)
    {
        if (m_params.find(key) == m_params.end())
        {
            m_error = key + " param error, param doesnt exist";
            return false;
        }

        try
        {
            val = std::stod(m_params[key]);
        }
        catch(const std::exception& e)
        {
            m_error = key + " param error, unable to convert to double: " + m_params[key];
            return false;
        }

        return true;
    }

    static void print_params()
    {
        std::cout << "ConfigFile Parameters:" << std::endl;
        for (const auto param: m_params)
        {
            std::cout << "  " << param.first << " = " << param.second << std::endl;
        }
    }
private:

    typedef std::map<std::string, std::string> ParamMap;

    ConfigFile() = default;
    virtual ~ConfigFile() = default;


    static bool set_filepath(int argc, const char** argv)
    {
        if (argc != 2)
        {
            m_error = "Usage: ./image_processor <path/to/config/file>";
            return false;
        }

        m_filepath = argv[1];
        if (!std::filesystem::is_regular_file(m_filepath))
        {
            m_error = "Config filepath is not a file: " + std::string(m_filepath);
            return false;
        }

        return true;
    }

    static bool parse_file()
    {
        std::ifstream config_file(m_filepath);
        std::string line = "";
        while (std::getline(config_file, line))
        {
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());

            if (line.empty())
            {
                continue;
            }

            if (line[0] == '#')
            {
                continue;
            }

            std::stringstream line_ss(line);
            std::string substring = "";
            std::vector<std::string> substrings;
            while(std::getline(line_ss, substring, '='))
            {
                if (!substring.empty())
                {
                    substrings.push_back(substring);
                }
            }
        
            if (substrings.size() != 2)
            {
                m_error = "Invalid config file line: " + line;
                return false;
            }

            if (m_params.find(substrings[0]) != m_params.end())
            {
                m_error = "Duplicate parameter found: " + substrings[0];
                return false;
            }
            m_params[substrings[0]] = substrings[1];
        }
        return true;
    }

    static bool extract_params()
    {
        bool display_type_found = false;
        m_delay = -1.0;
        for (const auto& param: m_params)
        {
            const std::string key = param.first;
            const std::string val = param.second;

            if (key == "INPUT_DIR")
            {
                if (!std::filesystem::is_directory(val))
                {
                    m_error = "INPUT_DIR param error, directory doesnt exist: " + val;
                    return false;
                }
                m_input_dir = val;
            }
            else if (key == "OUTPUT_DIR")
            {
                m_output_dir = val;
            }
            else if (key == "DISPLAY_TYPE")
            {
                for (size_t i = 0; i < DisplayTypeInt(DisplayType::SIZE); i++)
                {
                    if (val == DisplayTypeString((DisplayType) i))
                    {
                        m_display_type = (DisplayType) i;
                        display_type_found = true;
                    }
                }
            }
            else if (key == "DELAY")
            {
                try
                {
                    m_delay = std::stod(val);
                }
                catch(const std::exception& e)
                {
                    m_error = "DELAY param error, unable to convert to double: " + val;
                    return false;
                }
            }
        }
        if (m_input_dir.empty())
        {
            m_error = "INPUT_DIR param not found";
            return false;
        }
        if (m_output_dir.empty())
        {
            m_error = "OUTPUT_DIR param not found";
            return false;
        }
        if (!display_type_found)
        {
            m_error = "DISPLAY_TYPE param invalid or not found";
            return false;
        }
        if (m_delay < 0.0)
        {
            m_error = "DELAY param invalid or not found";
            return false;
        }
        return true;
    }

    inline static std::string m_error;
    inline static std::filesystem::path m_filepath;

    inline static std::filesystem::path m_input_dir;
    inline static std::filesystem::path m_output_dir;
    inline static DisplayType m_display_type;
    inline static double m_delay;

    inline static ParamMap m_params;
};

#endif /** CONFIGFILE_HPP_ */
