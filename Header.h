#ifndef __HEADER_H__
#define __HEADER_H__

/**
 *
 * @class  Header
 * @author ngrau@augie.edu
 * @date   January 2019
 * @brief  This is the header information that comes from the I-V and SPS data 
 *         from the LabView output of the SiPM tests
 *
 *         The following is the documentation (including versioning) of the 
 *         metadata associated with the LabView output of the I-V and SPS test
 *         for the sPHENIX SiPMs.
 *
 *         Operator name length [8-bit unsigned int]
 *         Operator name [array of 8-bit unsigned int]
 *         NI time stamp [double] (Note Epoch 12:00 AM, Jan 1, 1904 GMT)
 *         Tray ID length [8-bit unsigned int]
 *         Tray ID [array of 8-bit unsigned int]
 *         SiPM position in tray [8-bit unsigned]
 *
 */

#include <cstdint>
#include <cmath> 
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include "bithelper.h"
using namespace SiPM;

#ifdef __CINT__
typedef unsigned int uint8_t;
typedef unsigned int uint32_t;
typedef unsigned int uint64_t;
#endif

class Header {

 public:

  //!constructor -- input all information
  Header(uint8_t version, const std::string& op, uint64_t timestamp, const std::string& trayid, uint8_t position){
    _version = version;
    _operatorStringLength = op.size();
    _operator = new char[_operatorStringLength+1];
    for(uint8_t i=0; i<_operatorStringLength+1; i++){
      _operator[i] = op[i];
    }
    _timestamp = timestamp;
    _trayStringLength = trayid.size();
    _trayid = new char[_trayStringLength];
    for(uint8_t i=0; i<_operatorStringLength+1; i++){
      _trayid[i] = trayid[i];
    }
    _position = position;
  }

  //! constructor -- read in the header from the given file
  //! it is assumed the position in the file is at the beginning
  //! of the header
  Header(uint8_t version, std::fstream& file){
    _version = version;
    if(_version == 0) {
      _operatorStringLength = ReadData<uint8_t>(&file);
      _operator = new char[_operatorStringLength+1];
      file.read(_operator,_operatorStringLength);
      _operator[_operatorStringLength] = '\0';
      //the timestamp in the file uses the NI epoch of 12:00:00 AM GMT
      //January 1, 1904 to convert the epoch to UNIX standard (12:00:00 AM
      //GMT January 1, 1970) you must subtract 2082844800
      //but my tests of this do not work right now
      _timestamp = (uint64_t)ReadData<double>(&file);
      _timestamp -= 2082844800U;
      _trayStringLength = ReadData<uint8_t>(&file);
      _trayid = new char[_trayStringLength+1];
      file.read(_trayid,_trayStringLength);
      _trayid[_trayStringLength] = '\0';
      _position = ReadData<uint8_t>(&file);
    }
  }

  //! Copy constructor
  Header(const Header& h) {
    _version = h.Version();
    std::string op = h.Operator();
    _operatorStringLength = op.size();
    _operator = new char[_operatorStringLength+1];
    for(uint8_t i=0; i<_operatorStringLength+1; i++){
      _operator[i] = op[i];
    }
    _timestamp = h.TimeStamp();
    std::string trayid = h.TrayID();
    _trayStringLength = trayid.size();
    _trayid = new char[_trayStringLength];
    for(uint8_t i=0; i<_operatorStringLength+1; i++){
      _trayid[i] = trayid[i];
    }
    _position = h.Position();
  }

  //! Destructor
  virtual ~Header() {
    delete _operator;
    delete _trayid;
  }

  uint8_t Version() const {
    return _version;
  }

  //! get the operator as a c++ string
  std::string Operator() const {
    return std::string(_operator);
  }

  //! get the NI time stamp
  uint64_t TimeStamp() const {
    return _timestamp;
  }

  //! get the tray id barcode as a C++ string
  std::string TrayID() const {
    return std::string(_trayid);
  }

  //! get the linearized position of the SiPM within the tray
  //! this number will be in the range [0,168]
  uint8_t Position() const {
    return _position;
  }

  void Print() const {
    std::cout<<"Version: "<<(unsigned int)_version<<std::endl;
    std::cout<<"Operator: "<<Operator()<<std::endl;
    std::cout<<"UNIX timestamp: "<<(unsigned int)_timestamp<<std::endl;
    std::cout<<"Tray ID: "<<TrayID()<<std::endl;
    std::cout<<"Position: "<<(unsigned int)_position<<std::endl;
  }

  void Write(std::fstream* file){
    file->write((char*)&_operatorStringLength,1);
    file->write(_operator,_operatorStringLength);
    WriteData<uint64_t>(file,_timestamp);
    file->write((char*)&_trayStringLength,1);
    file->write(_trayid,_trayStringLength);
    file->write((char*)&_position,1);
  }

 private:
  uint8_t _version;
  uint8_t _operatorStringLength;
  char* _operator;
  uint64_t _timestamp;
  uint8_t _trayStringLength;
  char* _trayid;
  uint8_t _position;
  uint8_t _ntemp;
  double* _temp;

};

#endif /* __HEADER_H__ */
