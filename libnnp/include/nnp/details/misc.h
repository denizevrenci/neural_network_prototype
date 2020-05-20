#pragma once

namespace nnp {

namespace details {

template <typename ForwardIterator>
size_t argmax(ForwardIterator begin, ForwardIterator end)
{
	if (begin == end)
		return -1;
	ForwardIterator largest = begin;
	size_t maxIdx = 0;
	++begin;
	for (size_t curIdx = 1; begin != end; ++begin, ++curIdx)
		if (*largest < *begin)
		{
			largest = begin;
			maxIdx = curIdx;
		}
	return maxIdx;
}

} // namespace details

} // namespace nnp
