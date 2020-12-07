#ifndef CONFIGFILE_HPP_
#define CONFIGFILE_HPP_

#include <algorithm>
#include <fstream>
#include <map>
#include <vector>

#include "Logger.hpp"

/**
 * Provide a static ConfigFile class for the framework and all algorithms to use.
 */
class ConfigFile
{
public:

    /**
     * Initialize the config file with a filepath
     */
    static bool initialize(const path_t& filepath)
    {
        TRACE();

        // validate ConfigFile exists
        m_filepath = filepath;
        if (!fs::is_regular_file(m_filepath))
        {
            LOG(LogLevel::ERROR, "ConfigFile is not a file: ", m_filepath.string());
            return false;
        }
        LOG(LogLevel::DEBUG, "Found ConfigFile: ", m_filepath);

        // parse ConfigFile
        if (!parse_file())
            return false;
        LOG(LogLevel::DEBUG, "ConfigFile initialized");

        return true;
    }

    /**
     * Return a parameter with name=key.
     */
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
    inline static std::map<std::string, std::string> m_params;
};

/**
 * Template specializations for parameter types
 * (int, double, std::size_t, DisplayType)
 */

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
bool ConfigFile::get_param(const std::string& key, std::size_t& val)
{
    TRACE();

    if (!check_param_exists(key))
        return false;
    auto parser = [](const std::string& val_str) { return std::stoi(val_str); };

    int signed_val = 0;
    if (!ConfigFile::parse_number(m_params[key], signed_val, parser))
        return false;
    if (signed_val < 0)
    {
        LOG(LogLevel::ERROR, "Error getting parameter: ", key, " value must be positive");
        return false;
    }

    val = (std::size_t) signed_val;
    return true;
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
    if (!string_to_enum(m_params[key], val))
    {
        LOG(LogLevel::ERROR, "Invalid DISPLAY_TYPE value: ", m_params[key]);
        return false;
    }
    return true;
}

#endif /** CONFIGFILE_HPP_ */
