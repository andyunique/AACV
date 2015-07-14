#ifndef OSTIMER_H_
#define OSTIMER_H_

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h>

typedef struct timeval {
	long tv_sec;
	long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tz = NULL)
{
	static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

	SYSTEMTIME  system_time;
	FILETIME    file_time;
	uint64_t    time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}

#elif defined __unix__
#include <sys/time.h>
#endif



class OsTimer
{
private:
	struct timeval t0;
	struct timeval t1;

public:

OsTimer()
{
}

~OsTimer()
{
}

void start()
{
	gettimeofday(&t0 , NULL);
}

void stop()
{
	gettimeofday(&t1 , NULL);
}

double elapsedTime()
{
	return (t1.tv_sec-t0.tv_sec)*1000.0 + (t1.tv_usec-t0.tv_usec)/1000.0;
}

};


#endif

