#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <stdint.h>
#include <map>
#include <limits>
#include <string>

typedef uint16_t pixel16_t;
typedef uint64_t pixel64_t;
typedef std::map<std::string, std::string> params_t;

#define MAX_PIXEL_VAL 65535

#endif /** TYPES_HPP_ */