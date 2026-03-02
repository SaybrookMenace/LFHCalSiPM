//  test_reader.cpp
//  A sample C++ file for reading in SiPM data
//  A minimal working example for converting binary data to text file
//  Created by Ryan Hamilton on 11/3/25.

#include "test_reader.hpp"
#include "IVBinaryFile.h"


// main (this is always what g++ compiles and runs when executed)
// to compile, cd to this code in your terminal and type
//
//        g++ test_reader.cpp -o test_run
//
// This will make an executable file that can be run from terminal using
//
//        ./test_run
//
// which will execute the code in main(). The input parameters
// argc (argument count) and argv (argument vector) can be appended to the execution, e.g.
//
//        ./test_run 1 hello_world
//
// would pass in argc = 2 with argv[0] = "1" and argv[2] = "hello_world" as c_str type (i.e. char arrays)
// You could then use std::atoi() or std::atof() to convert the char array to string or float as you need.
// Iterating through many files could be done directly in C using the stat struct, or exernally using a
// bash script that passes in filenames as arguments to the compiled code in the above way.
//
// You can also modify the IVBinaryFile class if there are other data you are interested in saving from the binaries.
int main(int argc, char *argv[]) {
  // Instantiate the IV reader class with an output file (could loop over many files)
  IVBinaryFile* reader = new IVBinaryFile("test_data/IV_250717-1305_0_0.bin");
  
  // Make an outfile stream to send output to
  std::ofstream outfile("test_data/IV_250717_0_0.txt");
  
  // Redirect the std::cout buffer to your output file
  std::cout.rdbuf(outfile.rdbuf());
  
  // Print the data as read in by IVBinaryFile, the std::cout output gets piped to our new text file
  reader->Print();
  
  // Close the outstream and save the file
  outfile.close();
  
  // Reurn 0--success
  return 0;
}

