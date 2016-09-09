/*
 * memory_usages.cpp
 *
 *  Created on: 26-Nov-2014
 *      Author: amit
 */

#include "memory_usages.h"



long parseLine(char* line) {
	long i = strlen(line);
	while (*line < '0' || *line > '9')
		line++;
	line[i - 3] = '\0';
	i = atoi(line);
	return i;
}

/*
 * Returns the Virtual Memory currently used by current process
 *
 * Actually, virtual memory = part in RAM + part on disk + virtual address space not mapped to physical memory
 *  + memory mapped files + shared memory.
 * So to measure memory consumption, VmSize is pretty useless.
 *
 */
long getCurrentProcess_VirtualMemoryUsed() { //Note: this value is in KB!
	FILE* file = fopen("/proc/self/status", "r");	//reading the Linux resource file
	long result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, "VmSize:", 7) == 0) {		//VmSize : Virtual Memory Size currently used
			result = parseLine(line);
			break;
		}
	}
	fclose(file);
	return result;
}

/*
 * Returns the Physical Memory currently used by current process:
 */
long getCurrentProcess_PhysicalMemoryUsed() { //Note: this value is in KB!
	FILE* file = fopen("/proc/self/status", "r");	//reading the Linux resource file
	long result = -1;
	char line[128];

	while (fgets(line, 128, file) != NULL) {
		if (strncmp(line, "VmRSS:", 6) == 0) {		//VmRSS : Virtual Memory Size currently used
			result = parseLine(line);
			break;
		}
	}
	fclose(file);
	return result;
}
