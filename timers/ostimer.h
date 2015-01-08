#ifndef OSTIMER_H_
#define OSTIMER_H_

#include <sys/time.h>

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