#ifndef __BITHELPER_H__
#define __BITHELPER_H__

/**
 *  @namespace SiPM
 *  @author    ngrau@augie.edu
 *  @date      February 2019
 *  @brief     Helper functions to undo endianess of multi-byte output from 
 *             NI. This was necessary for my Mac-OS 10.14 computer and may 
 *             or may not be necessary on other computers. Best to define a 
 *             compiler flag that will act appropriately.
 *
 */

#include <fstream>

namespace SiPM {

  template <typename T>
  inline T ReadData(std::fstream* f){
    unsigned short nbytes = sizeof(T);
    unsigned char *n = new unsigned char[nbytes];
    T ret;
    unsigned char *pret = (unsigned char*)&ret;
    for(unsigned short i=0; i<nbytes; i++){
      f->read((char*)&n[i],1);
      //pret[nbytes-1-i] = n[i];
	  pret[i] = n[i];   //remove endian conversion!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    }
    delete[] n;
    return ret;
  }

  template <typename T>
  inline void WriteData(std::fstream* f, T data){
    unsigned short nbytes = sizeof(T);
    unsigned char *pdata = (unsigned char*)&data;
    for(short i=nbytes-1; i>=0; i--){
      f->write((char*)&pdata[i],1);
    }
  }

};

#endif /* __BITHELPER_H__ */
