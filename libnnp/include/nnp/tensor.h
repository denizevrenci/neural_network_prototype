#pragma once

#include <dlib/matrix/matrix.h>

#include "common.h"

namespace nnp {

template <typename Float = float, size_t SIZE = RESIZEABLE, size_t BATCH_SIZE = RESIZEABLE>
class Tensor {
public:
	using Data = dlib::matrix<Float, SIZE, BATCH_SIZE, dlib::default_memory_manager, dlib::column_major_layout>;

	Tensor() = default; // TODO

	template <size_t SIZE_D = SIZE, size_t BATCH_SIZE_D = BATCH_SIZE
		, std::enable_if_t<SIZE_D == RESIZEABLE && BATCH_SIZE_D != RESIZEABLE>* = nullptr>
	Tensor(size_t size)
		: m_data(size, BATCH_SIZE) {
	}

	template <size_t SIZE_D = SIZE, size_t BATCH_SIZE_D = BATCH_SIZE
		, std::enable_if_t<SIZE_D != RESIZEABLE && BATCH_SIZE_D == RESIZEABLE>* = nullptr>
	Tensor(size_t batchSize)
		: m_data(SIZE, batchSize) {
	}

	template <size_t SIZE_D = SIZE, size_t BATCH_SIZE_D = BATCH_SIZE
		, std::enable_if_t<SIZE_D == RESIZEABLE && BATCH_SIZE_D == RESIZEABLE>* = nullptr>
	Tensor(size_t size, size_t batchSize)
		: m_data(size, batchSize) {
	}

	Tensor(const Data& data)
		: m_data(data) {
	}

	Tensor(Data&& data)
		: m_data(std::move(data)) {
	}

	~Tensor() {}

	decltype(auto) begin() { return m_data.begin(); }

	decltype(auto) begin() const { return m_data.begin(); }

	decltype(auto) end() { return m_data.end(); }

	decltype(auto) end() const { return m_data.end(); }

	decltype(auto) operator ()(size_t row, size_t column) {
		return m_data(row, column);
	}

	decltype(auto) operator ()(size_t row, size_t column) const {
		return m_data(row, column);
	}

	size_t size() const { return m_data.nr(); }

	size_t batchSize() const { return m_data.nc(); }

	template <size_t SIZE_D = SIZE
		, typename = std::enable_if_t<SIZE_D == RESIZEABLE>>
	void setSize(size_t newSize) {
		m_data.set_size(newSize, m_data.nc());
	}

	template <size_t BATCH_SIZE_D = SIZE
		, typename = std::enable_if_t<BATCH_SIZE_D == RESIZEABLE>>
	void setBatchSize(size_t newSize) {
		m_data.set_size(m_data.nr(), newSize);
	}

	Data& data() { return m_data; }

	const Data& data() const { return m_data; }

private:
	Data m_data;
};

} // namespace nnp
