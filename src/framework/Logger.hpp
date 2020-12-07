#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include <iostream>

#include "Utils.hpp"

/** Define logging macros. */
#define LOG(level, ...) { Logger::_log(__FILE__, __LINE__, level, __VA_ARGS__); }
#define TRACE() { Logger::_log(__FILE__, __LINE__, LogLevel::TRACE, __func__, "()"); }

/**
 * Provide a logger for the framework.
 */
class Logger
{
public:

    /** Initialize the logger with a LogLevel */
    static void initialize(LogLevel level) { m_level = level; }

    /**
     * Print a log message, this function should not be called directly,
     * instead the above macros should be used.
     */
    template<typename ...Args>
    static void _log(const char* file, int line, LogLevel level, Args && ...args)
    {
        if (level <= m_level)
        {
            (std::cout << std::left << std::setw(5) << magic_enum::enum_name(level) << " - "
                       << file << ":" << line << " - " << ... << args) << std::endl;
        }
    }

private:

    inline static LogLevel m_level = LogLevel::ERROR;
};

#endif /** LOGGER_HPP_ */
