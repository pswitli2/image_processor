#ifndef BASEIMAGEALGORITHM_HPP_
#define BASEIMAGEALGORITHM_HPP_

#include "Types.hpp"

class BaseImageAlgorithm
{
public:

    ~BaseImageAlgorithm()
    {
    }

    virtual std::string name() = 0;

    virtual bool initialize() = 0;

    virtual bool update(const png::image<pixel_t>& input, png::image<pixel_t>& output) = 0;

protected:

    BaseImageAlgorithm()
    {
    }

private:
};

typedef std::shared_ptr<BaseImageAlgorithm> BaseImageAlgorithmPtr;
typedef std::vector<BaseImageAlgorithmPtr> BaseImageAlgorithmPtrVec;
#endif /** BASEIMAGEALGORITHM_HPP_ */