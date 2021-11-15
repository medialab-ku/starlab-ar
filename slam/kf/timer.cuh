#pragma once

#include <iostream>
#include <chrono>
#include <string>

namespace kf
{
	typedef std::chrono::steady_clock TClock;
	typedef std::chrono::time_point<TClock> TTime;
	class CTimer
	{
	 private:
		std::string m_strName;
		TTime m_timeStart, m_timeEnd;

	 public:
		CTimer() = delete;
		CTimer(std::string strName);
		~CTimer();

	 private:
		void start();
		void end();
		void print() const;
	};
}
