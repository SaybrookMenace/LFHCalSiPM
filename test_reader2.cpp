//  test_reader2.cpp
//  Iterate through IV files: IV_250821-1301_X1_X2.bin
//  X1 = 0..22 , X2 = 0..19

#include "test_reader.hpp"
#include "IVBinaryFile.h"
#include <sstream>
#include <iomanip>

int main(int argc, char *argv[]) {

    for (int x1 = 0; x1 <= 22; x1++) {
        for (int x2 = 0; x2 <= 19; x2++) {

            // Construct binary input filename
            std::ostringstream infile_name;
            infile_name << "test_data/IV_250821-1301_" << x1 << "_" << x2 << ".bin";

            // Construct text output filename
            std::ostringstream outfile_name;
            outfile_name << "test_data/IV_250821-1301_" << x1 << "_" << x2 << ".txt";

            // Instantiate reader for this file
            IVBinaryFile* reader = new IVBinaryFile(infile_name.str().c_str());

            // Open output text file
            std::ofstream outfile(outfile_name.str().c_str());

            // Redirect stdout to outfile
            std::streambuf* old_buf = std::cout.rdbuf(outfile.rdbuf());

            // Print data from binary file into text file
            reader->Print();

            // Restore normal std::cout behavior
            std::cout.rdbuf(old_buf);

            outfile.close();
            delete reader;
        }
    }

    return 0;
}