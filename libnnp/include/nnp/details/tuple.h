#pragma once

#include <tuple>
#include <utility>

namespace nnp {

namespace impl {

namespace details {

template <typename Tuple>
using TupleSequence =
	std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value>;

template <typename Tuple, typename Callable>
constexpr void forEachHelper(const Tuple&, Callable&& c, std::index_sequence<>) {}

template <typename Tuple, typename Callable, size_t HEAD, size_t... TAIL>
constexpr void
	forEachHelper(const Tuple& tuple, Callable&& c, std::index_sequence<HEAD, TAIL...>)
{
	c(std::get<HEAD>(tuple));
	forEachHelper(tuple, std::forward<Callable>(c), std::index_sequence<TAIL...>());
}

template <typename Tuple, typename Callable>
constexpr void forEachHelper(Callable&& c, std::index_sequence<>) {}

template <typename Tuple, typename Callable, size_t HEAD, size_t... TAIL>
constexpr void forEachHelper(Callable&& c, std::index_sequence<HEAD, TAIL...>)
{
	c.template operator()<typename std::tuple_element_t<HEAD, Tuple>>();
	forEachHelper<Tuple, Callable>(std::forward<Callable>(c), std::index_sequence<TAIL...>());
}

} // namespace details

template <typename Tuple, typename Callable>
constexpr void forEach(Tuple&& tuple, Callable&& c)
{
	details::forEachHelper(tuple, std::forward<Callable>(c), details::TupleSequence<Tuple>());
}

template <typename Tuple, typename Callable>
constexpr void forEach(Callable&& c)
{
	details::forEachHelper<Tuple>(std::forward<Callable>(c), details::TupleSequence<Tuple>());
}

} // namespace impl

} // namespace nnp
