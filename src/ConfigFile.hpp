#ifndef CONFIGFILE_HPP_
#define CONFIGFILE_HPP_

#include <algorithm>
#include <map>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "Logger.hpp"

class ConfigFile
{
public:

    static bool initialize(const path_t& filepath)
    {
        TRACE();

        m_filepath = filepath;

        // validate ConfigFile exists
        if (!fs::is_regular_file(m_filepath))
        {
            LOG(LogLevel::ERROR, "ConfigFile is not a file: ", m_filepath.string());
            return false;
        }
        LOG(LogLevel::DEBUG, "Found ConfigFile: ", m_filepath);

        // parse ConfigFile
        if (!parse_file())
            return false;

        return true;
    }

    template<class T>
    static bool get_param(const std::string& key, T& val)
    {
        TRACE();

        if (!check_param_exists(key))
            return false;

        val = m_params[key];
        return true;
    }

private:

    ConfigFile() = default;
    virtual ~ConfigFile() = default;

    static bool parse_file()
    {
        TRACE();

        std::ifstream config_file(m_filepath);
        std::string line = "";

        // loop over lines
        while (std::getline(config_file, line))
        {
            // remove whitespace from line
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());

            // skip empty lines
            if (line.empty())
                continue;

            // skip lines beginning with #
            if (line[0] == '#')
                continue;

            // split line by = signs
            std::stringstream line_ss(line);
            std::string part = "";
            std::vector<std::string> parts;
            while(std::getline(line_ss, part, '='))
                if (!part.empty())
                    parts.push_back(part);
        
            // if more than 2 parts, invalid line
            if (parts.size() != 2)
            {
                LOG(LogLevel::ERROR, "Invalid ConfigFile, error in line: ", line);
                return false;
            }

            // check for duplicate parameter
            const auto& key = parts[0];
            const auto& val = parts[1];
            if (m_params.find(key) != m_params.end())
            {
                LOG(LogLevel::ERROR, "Invalid ConfigFile, Duplicate parameter found: ", key);
                return false;
            }

            LOG(LogLevel::DEBUG, "Found ConfigFile parameter: ", key, " = ", val);

            // save parameter
            m_params[key] = val;
        }
        return true;
    }

    static bool check_param_exists(const std::string& key)
    {
        TRACE();

        if (m_params.find(key) == m_params.end())
        {
            LOG(LogLevel::ERROR, "Parameter with key: ", key, " not found");
            return false;
        }
        return true;
    }

    template<typename T, typename F>
    static bool parse_number(const std::string& val_str, T& val, F parser)
    {
        TRACE();

        try
        {
            val = parser(val_str);
        }
        catch(const std::exception& e)
        {
            LOG(LogLevel::ERROR, "Error getting parameter, unable to convert '", val_str, "' to number, error: ", e.what());
            return false;
        }

        return true;
    }

    inline static path_t m_filepath;
    inline static params_t m_params;
};


template<>
bool ConfigFile::get_param(const std::string& key, int& val)
{
    TRACE();
    if (!check_param_exists(key))
        return false;
    auto parser = [](const std::string& val_str) { return std::stoi(val_str); };
    return ConfigFile::parse_number(m_params[key], val, parser);
}

template<>
bool ConfigFile::get_param(const std::string& key, double& val)
{
    TRACE();
    if (!check_param_exists(key))
        return false;
    auto parser = [](const std::string& val_str) { return std::stod(val_str); };
    return ConfigFile::parse_number(m_params[key], val, parser);
}

template<>
bool ConfigFile::get_param(const std::string& key, DisplayType& val)
{
    TRACE();
    if (!check_param_exists(key))
        return false;
    return string_to_enum(m_params[key], val);
}

#endif /** CONFIGFILE_HPP_ */

    // static bool extract_params()
    // {
    //     bool display_type_found = false;
    //     bool log_level_found = false;
    //     m_delay = -1.0;
    //     for (const auto& param: m_params)
    //     {
    //         const std::string key = param.first;
    //         const std::string val = param.second;

    //         if (key == "INPUT_DIR")
    //         {
    //             if (!std::filesystem::is_directory(val))
    //             {
    //                 m_error = "INPUT_DIR param error, directory doesnt exist: " + val;
    //                 return false;
    //             }
    //             m_input_dir = val;
    //         }
    //         else if (key == "OUTPUT_DIR")
    //         {
    //             m_output_dir = val;
    //         }
    //         else if (key == "DISPLAY_TYPE")
    //         {
    //             for (std::size_t i = 0; i < DisplayTypeInt(DisplayType::SIZE); i++)
    //             {
    //                 if (val == DisplayTypeString((DisplayType) i))
    //                 {
    //                     m_display_type = (DisplayType) i;
    //                     display_type_found = true;
    //                 }
    //             }
    //         }
    //         else if (key == "LOG_LEVEL")
    //         {
    //             for (std::size_t i = 0; i < LogLevelInt(LogLevel::SIZE); i++)
    //             {
    //                 if (val == LogLevelString((LogLevel) i))
    //                 {
    //                     m_log_level = (LogLevel) i;
    //                     log_level_found = true;
    //                 }
    //             }
    //         }
    //         else if (key == "DELAY")
    //         {
    //             try
    //             {
    //                 m_delay = std::stod(val);
    //             }
    //             catch(const std::exception& e)
    //             {
    //                 m_error = "DELAY param error, unable to convert to double: " + val;
    //                 return false;
    //             }
    //         }
    //     }
    //     if (m_input_dir.empty())
    //     {
    //         m_error = "INPUT_DIR param not found";
    //         return false;
    //     }
    //     if (m_output_dir.empty())
    //     {
    //         m_error = "OUTPUT_DIR param not found";
    //         return false;
    //     }
    //     if (!display_type_found)
    //     {
    //         m_error = "DISPLAY_TYPE param invalid or not found";
    //         return false;
    //     }
    //     if (!log_level_found)
    //     {
    //         m_error = "LOG_LEVEL param invalid or not found";
    //         return false;
    //     }
    //     if (m_delay < 0.0)
    //     {
    //         m_error = "DELAY param invalid or not found";
    //         return false;
    //     }
    //     return true;
    // }

    // inline static std::filesystem::path m_input_dir;
    // inline static std::filesystem::path m_output_dir;
    // inline static DisplayType m_display_type;
    // inline static LogLevel m_log_level;
    // inline static double m_delay;

    // static const std::filesystem::path input_dir() { return m_input_dir; }
    // static const std::filesystem::path output_dir() { return m_output_dir; }
    // static DisplayType display_type() { return m_display_type; }
    // static LogLevel log_level() { return m_log_level; }
    // static double delay() { return m_delay; }
    // static const std::string error() { return m_error; }

    // static int get_int_param(const std::string& key, int& val)
    // {
    //     if (m_params.find(key) == m_params.end())
    //     {
    //         m_error = key + " param error, param doesnt exist";
    //         return false;
    //     }

    //     try
    //     {
    //         val = std::stoi(m_params[key]);
    //     }
    //     catch(const std::exception& e)
    //     {
    //         m_error = key + " param error, unable to convert to int: " + m_params[key];
    //         return false;
    //     }

    //     return true;
    // }

    // static bool get_double_param(const std::string& key, double& val)
    // {
    //     if (m_params.find(key) == m_params.end())
    //     {
    //         m_error = key + " param error, param doesnt exist";
    //         return false;
    //     }

    //     try
    //     {
    //         val = std::stod(m_params[key]);
    //     }
    //     catch(const std::exception& e)
    //     {
    //         m_error = key + " param error, unable to convert to double: " + m_params[key];
    //         return false;
    //     }

    //     return true;
    // }