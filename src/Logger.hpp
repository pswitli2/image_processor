#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include <iostream>
#include <utility>

#include "Types.hpp"

enum class LogLevel
{
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE,
    SIZE
};

static inline size_t LogLevelInt(const LogLevel& log_level)
{
    return static_cast<size_t>(log_level);
}

static inline std::string LogLevelString(const LogLevel& level)
{
    switch (level)
    {
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::WARN: return "WARN";
        case LogLevel::INFO: return "INFO";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::SIZE: return "SIZE";
        default: return "";
    }
}

class Logger
{
public:

    static void initialize(LogLevel level) { m_level = level; }

    template<typename ...Args>
    static void log(LogLevel level, Args && ...args)
    {
        if (level <= m_level)
        {
            (std::cout << std::setw(5) << LogLevelString(level) << ": " << ... << args) << std::endl;
        }
    }

private:

    inline static LogLevel m_level;
};

#endif /** LOGGER_HPP_ */
