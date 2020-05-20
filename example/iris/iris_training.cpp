#include <iomanip>
#include <iostream>
#include <random>

#include <nnp/details/misc.h>
#include <nnp/loss.h>
#include <nnp/network.h>

#include "dataset.h"

template <typename Float>
class NormalDistGenerator
{
public:
	Float operator()() { return m_dis(m_gen); }

private:
	std::mt19937 m_gen;
	std::normal_distribution<Float> m_dis;
};

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		std::cout << "Usage: " << argv[0] << " <dataset path>\n";
		return 1;
	}

	NormalDistGenerator<float> gen;

	dset::Data data(argv[1]);

	using BaseNetwork =
		nnp::TupleNetwork<nnp::SigmoidLayer<float, 15, 4>, nnp::LinearLayer<float, 3, 15>>;

	BaseNetwork baseNetwork{
		nnp::SigmoidLayer<float, 15, 4>{gen}, nnp::LinearLayer<float, 3, 15>(gen)};

	using TrainingNetwork = nnp::Network<BaseNetwork&, nnp::SoftMaxLayer<float>>;

	TrainingNetwork trainingNetwork{baseNetwork, nnp::SoftMaxLayer<float>{}};

	std::cout << "Epoch      Training loss  Validation loss  Test accuracy" << std::endl;
	std::cout << std::fixed << std::setprecision(5);
	for (size_t ii = 0; ii != 15001; ++ii)
	{
		auto loss = trainingNetwork.propagate(
			data.trainingInput(), data.trainingCrossVal(), 0.002f, 5e-5f);
		if (ii % 100 == 0)
		{
			auto out = baseNetwork.forward(data.testInput());
			size_t matchCount = 0;
			for (size_t jj = 0; jj != out.batchSize(); ++jj)
				matchCount +=
					nnp::details::argmax(&out(0, jj), &out(0, jj + 1)) ==
					nnp::details::argmax(
						&data.testCrossVal()(0, jj), &data.testCrossVal()(0, jj + 1));
			std::cout
				<< std::setw(8) << ii << std::setw(10) << loss << std::setw(15)
				<< trainingNetwork.propagate(
					data.validationInput(), data.validationCrossVal(), 5e-5f)
				<< std::setw(17) << static_cast<double>(matchCount) / out.batchSize()
				<< std::endl;
		}
	}
}
