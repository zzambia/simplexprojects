/*
 * memory_usages.h
 *
 *  Created on: 26-Nov-2014
 *      Author: amit
 *      Refrence : http://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
 *
 */

#ifndef MEMORY_USAGES_H_
#define MEMORY_USAGES_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>


long parseLine(char* line);

/*
 * Returns the Virtual Memory currently used by current process
 */
long getCurrentProcess_VirtualMemoryUsed();


/*
 * Returns the Physical Memory currently used by current process:
 */
long getCurrentProcess_PhysicalMemoryUsed();

#endif /* MEMORY_USAGES_H_ */
