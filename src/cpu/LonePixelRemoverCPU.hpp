#ifndef LONEPIXELREMOVER_HPP_
#define LONEPIXELREMOVER_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

/**
 * The LonePixelRemover algorithm filters out pixels that are surrounded by
 * non bright pixels. If a single pixel is bright, and all adjacent pixels arent,
 * its likely not a valid detection.
 *
 * Returied Parameters
 *     LONE_PIXEL_NUM_ADJACENT = Max number of adjacent bright pixels to be considered a "Lone pixel"
 */
class LonePixelRemoverCPU: public BaseImageAlgorithm
{
public:

    LonePixelRemoverCPU() = default;

    ~LonePixelRemoverCPU() override = default;

    std::string name() const override { return "LonePixelRemoverCPU"; }

    bool initialize_impl() override
    {
        TRACE();

        if (!ConfigFile::get_param(LONE_PIXEL_NUM_ADJACENT_PARAM_NAME, m_num_adjacent))
            return false;

        return true;
    }

    bool update_impl(const pixel64_t* input, pixel64_t* output) override
    {
        TRACE();

        // copy input to output
        memcpy(output, input, size_bytes());

        // loop over pixels
        std::vector<std::size_t> surrounding(8);
        for (std::size_t i = 1; i < width() - 1; i++)
        {
            for (std::size_t j = 1; j < height() - 1; j++)
            {
                // sum surrounding pixel values
                surrounding.clear();
                const std::size_t idx = i * height() + j;
                surrounding.push_back(idx + 1);
                surrounding.push_back(idx - 1);
                surrounding.push_back(idx - width());
                surrounding.push_back(idx + width());
                surrounding.push_back(surrounding[3] - 1);
                surrounding.push_back(surrounding[3] + 1);
                surrounding.push_back(surrounding[4] - 1);
                surrounding.push_back(surrounding[4] + 1);
                pixel64_t sum = 0;
                for (const auto& pixel_idx: surrounding)
                {
                    sum += input[pixel_idx];
                }

                // filter out pixel if not enough bright pixels around it
                if (sum <= MAX_PIXEL_VAL * m_num_adjacent)
                    output[idx] = 0;
            }
        }
        
        return true;
    }

protected:
private:

    std::size_t m_num_adjacent;
};


#endif /** LONEPIXELREMOVER_HPP_ */
