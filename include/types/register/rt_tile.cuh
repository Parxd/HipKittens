/**
 * @file
 * @brief Layouts and their manipulations for register tiles.
 */

#pragma once

#include <concepts>

namespace kittens {
namespace ducks {
/**
* @namespace rt_tile
* 
* @brief A namespace for template metaprogramming with register tile layouts.
* Assumption below is that the col is the reduction dimension
*/
namespace rt_tile {
 
template<int _rows, int _cols, int _stride>
struct rt_tile {
    static constexpr int rows = _rows;
    static constexpr int cols = _cols;
    static constexpr int stride = _stride;
    static constexpr int num_elements = rows*cols;
    static constexpr int elements_per_thread = num_elements / kittens::WARP_THREADS;
};

using 16x16 = rt_tile<16, 16, 4>;
using 32x32 = rt_tile<32, 32, 4>;
using 16x32 = rt_tile<16, 32, 8>;
using 32x16 = rt_tile<32, 16, 8>;

 
template<typename T>
concept all = std::is_same_v<T, 16x16> || std::is_same_v<T, 32x32> || std::is_same_v<T, 16x32> || std::is_same_v<T, 32x16>;

} // namespace rt_tile
} // namespace ducks
} // namespace kittens