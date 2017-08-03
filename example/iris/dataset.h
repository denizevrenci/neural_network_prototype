#pragma once

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include <libnnp/tensor.h>

namespace dset {

namespace details {

constexpr double TEST_RATIO = 0.25;
constexpr double VALIDATION_RATIO = (1 - TEST_RATIO) * 0.25;
constexpr double TRAIN_RATIO = 1 - TEST_RATIO - VALIDATION_RATIO;
constexpr size_t DATASET_SIZE = 150;
constexpr size_t TRAIN_SET_SIZE = DATASET_SIZE * TRAIN_RATIO;
constexpr size_t VALIDATION_SET_SIZE = DATASET_SIZE * VALIDATION_RATIO;
constexpr size_t TEST_SET_SIZE = DATASET_SIZE * TEST_RATIO;

enum class IrisFlower : uint8_t {
	SETOSA,
	VERSICOLOR,
	VIRGINICA
};

IrisFlower toIrisFlower(const char* str) {
	constexpr char SETOSA_STR[] = "Iris-setosa";
	constexpr char VERSICOLOR_STR[] = "Iris-versicolor";
	constexpr char VIRGINICA_STR[] = "Iris-virginica";
	if (std::strncmp(str, SETOSA_STR, sizeof(SETOSA_STR) - 1) == 0)
		return IrisFlower::SETOSA;
	else if (std::strncmp(str, VERSICOLOR_STR, sizeof(VERSICOLOR_STR) - 1) == 0)
		return IrisFlower::VERSICOLOR;
	else if (std::strncmp(str, VIRGINICA_STR, sizeof(VIRGINICA_STR) - 1) == 0)
		return IrisFlower::VIRGINICA;
	else
		throw std::runtime_error("Failed parsing iris flower type");
}

struct IrisData {
	std::array<float, 4> feat;
	IrisFlower type;
};

std::array<IrisData, DATASET_SIZE> loadIrisData(const char* filePath) {
	std::array<IrisData, DATASET_SIZE> data;
	std::ifstream irisFile(filePath);
	char buf[16]; // Length of longest string expected.
	for (size_t ii = 0; ii != DATASET_SIZE; ++ii) {
		auto& datum = data[ii];
		char** dummy = nullptr;
		for (size_t ii = 0; ii != 4; ++ii) {
			irisFile.getline(buf, sizeof(buf), ',');
			datum.feat[ii] = std::strtof(buf, dummy);
		}
		irisFile.getline(buf, sizeof(buf));
		datum.type = toIrisFlower(buf);
	}
	return data;
}

} // namespace details

class Data {
public:
	explicit Data(const char* filePath) {
		auto data = details::loadIrisData(filePath);
		std::shuffle(data.begin(), data.end(), std::mt19937{});
		fillDataSet(data, m_trainingInput, m_trainingCrossVal, 0);
		fillDataSet(data, m_validationInput, m_validationCrossVal, details::TRAIN_SET_SIZE);
		fillDataSet(data, m_testInput, m_testCrossVal
			, details::TRAIN_SET_SIZE + details::VALIDATION_SET_SIZE);
	}

	const auto& trainingInput() const { return m_trainingInput; }

	const auto& trainingCrossVal() const { return m_trainingCrossVal; }

	const auto& validationInput() const { return m_validationInput; }

	const auto& validationCrossVal() const { return m_validationCrossVal; }

	const auto& testInput() const { return m_testInput; }

	const auto& testCrossVal() const { return m_testCrossVal; }

private:
	nnp::Tensor<float, 4, details::TRAIN_SET_SIZE> m_trainingInput;
	nnp::Tensor<float, 3, details::TRAIN_SET_SIZE> m_trainingCrossVal;
	nnp::Tensor<float, 4, details::VALIDATION_SET_SIZE> m_validationInput;
	nnp::Tensor<float, 3, details::VALIDATION_SET_SIZE> m_validationCrossVal;
	nnp::Tensor<float, 4, details::TEST_SET_SIZE> m_testInput;
	nnp::Tensor<float, 3, details::TEST_SET_SIZE> m_testCrossVal;

	template <size_t SIZE>
	void fillDataSet(const std::array<details::IrisData, details::DATASET_SIZE>& data
		, nnp::Tensor<float, 4, SIZE>& input
		, nnp::Tensor<float, 3, SIZE>& crossVal
		, size_t offset) {
		for (size_t ii = 0; ii != SIZE; ++ii) {
			const auto& datum = data[ii + offset];
			for (size_t jj = 0; jj != 4; ++jj)
				input(jj, ii) = datum.feat[jj];
			for (size_t jj : {0, 1, 2})
				crossVal(jj, ii)
					= jj == (size_t)datum.type ? 1.f : 0.f;
		}
	}
};

} // namespace dset
