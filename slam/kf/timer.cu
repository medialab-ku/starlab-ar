#include "timer.cuh"

namespace kf
{

	CTimer::CTimer(std::string strName) :
		m_strName(strName)
	{
		start();
	}

	CTimer::~CTimer()
	{
		end();

		print();
	}

	void CTimer::start()
	{
		// set start time point
		m_timeStart = TClock::now();
	}

	void CTimer::end()
	{
		// set end time point
		m_timeEnd = TClock::now();
	}

	void CTimer::print() const
	{
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
			m_timeEnd - m_timeStart).count();
		std::cout << m_strName << " - " << duration << "ms" << std::endl;
	}
}
