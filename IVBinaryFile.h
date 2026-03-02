#ifndef __IVBINARYFILE_H__
#define __IVBINARYFILE_H__


#include <fstream>
#include <iostream>
#include "Header.h"
#include "bithelper.h"
using namespace SiPM;

#ifdef __CINT__
typedef unsigned int uint8_t;
typedef unsigned int uint32_t;
typedef unsigned int uint64_t;
#endif

class IVBinaryFile {

 public:

  //!constructor -- read in the contents of filename into memory
  IVBinaryFile(const char* filename){
    _file = new std::fstream(filename,std::ios::in | std::ios::binary);
    uint8_t version = ReadData<uint8_t>(_file);
	
	_TrayIDStringLength = ReadData<uint8_t>(_file);  //todo remove duplicate from binary file
	_TrayIDStringLength = ReadData<uint8_t>(_file);
    _TrayID = new char[_TrayIDStringLength+1];
    _file->read(_TrayID,_TrayIDStringLength);
    _TrayID[_TrayIDStringLength] = '\0';
	
	_SipmIDStringLength = ReadData<uint8_t>(_file);   //todo remove duplicate from binary file
	_SipmIDStringLength = ReadData<uint8_t>(_file);
    _Sipm_ID_String = new char[_SipmIDStringLength+1];
    _file->read(_Sipm_ID_String,_SipmIDStringLength);
    _Sipm_ID_String[_SipmIDStringLength] = '\0';
	
	//_module = ReadData<uint8_t>(_file);
	//_SIPM_ID = ReadData<uint8_t>(_file);
	_Socket = ReadData<uint8_t>(_file); 
	
	printf("version:%d %d %s %d %s socket_%d\n", version, _TrayIDStringLength,_TrayID, _SipmIDStringLength, _Sipm_ID_String, _Socket);
	
	_timestamp = ReadData<uint64_t>(_file);
    //_timestamp -= 2082844800U;
	_DMM_resistance = ReadData<double>(_file);
	printf("TS:%llu, dmm:%f\n", _timestamp, _DMM_resistance); 
	
	_ntemp = 10;
    _temps_initial = new double[_ntemp];
    _temps_final = new double[_ntemp];
	for(uint32_t i=0; i<_ntemp; i++){
      _temps_initial[i] = ReadData<double>(_file);
    }
    for(uint32_t i=0; i<_ntemp; i++){
      _temps_final[i] = ReadData<double>(_file);
    }
	
	
    _nfinemeasurements = ReadData<uint32_t>(_file);  
    _finevoltage = new double[_nfinemeasurements];
    _finecurrent = new double[_nfinemeasurements];
	_finevoltageSMU = new double[_nfinemeasurements];
	
	//printf("nmeas %d\n", _nfinemeasurements);
    for(uint32_t i=0; i<_nfinemeasurements; i++){
      _finevoltage[i] = ReadData<double>(_file);
    }
	for(uint32_t i=0; i<_nfinemeasurements; i++){
     _finecurrent[i] = ReadData<double>(_file);
    }
	for(uint32_t i=0; i<_nfinemeasurements; i++){
      _finevoltageSMU[i] = ReadData<double>(_file);  
    }
	
	//dark current measurement data
	_Idark0 = ReadData<double>(_file);  //Vbr-5V
	_Idark1 = ReadData<double>(_file);  //Vbr+3V
	_Ileakage0 = ReadData<double>(_file);
	_Ileakage1 = ReadData<double>(_file);
	_IdarkStartTemp = ReadData<double>(_file);
	_IdarkEndTemp = ReadData<double>(_file);
	
	//forward_resistance measurement data
	_forward_Resistance = ReadData<double>(_file);
	_forward_Res_V1 = ReadData<double>(_file);
	_forward_Res_V2 = ReadData<double>(_file);
	_forward_Res_I1 = ReadData<double>(_file);
	_forward_Res_I2 = ReadData<double>(_file);
	_forward_Res_StartTemp = ReadData<double>(_file);
	_forward_Res_StartTemp = ReadData<double>(_file);
	
	
	//DMM resistance compensation
	
	for(int i=0;i<_nfinemeasurements;i++)
	{
		_finecurrent[i] = _finecurrent[i] - (_finevoltage[i]/_DMM_resistance);
		if(_finecurrent[i] < 0.0) 
			_finecurrent[i] = 0.00000;
		//printf("%d curr:%lf\n", i, _finecurrent[i]);
	}
	
	
	
    

    _file->close();
  }


  //! destructor
  virtual ~IVBinaryFile(){
    delete _file;
    delete _finevoltage;
    delete _finecurrent;
    delete _temps_initial;
    delete _temps_final;
  }





  //! the number of fine measurements
  uint32_t NFineMeasurements() const {
    return _nfinemeasurements;
  }

  //! the array of voltage steps in the fine measurements
  double* FineVoltage() const {
    double *ret = new double[_nfinemeasurements];
    for(uint32_t i=0; i<_nfinemeasurements; i++){
      ret[i] = _finevoltage[i];
    }
    return ret;
  }

  //! the array of currents in the fine measurements
  double* FineCurrent() const {
    double *ret = new double[_nfinemeasurements];
    for(uint32_t i=0; i<_nfinemeasurements; i++){
      ret[i] = _finecurrent[i];
    }
    return ret;
  }

  //! a particular voltage in the fine measurements
  //! if requested bin out of range, garbage -9999 is returned
  double FineVoltage(uint32_t i) const {
    if(i<_nfinemeasurements)
      return _finevoltage[i];
    return -9999.;
  }

  //! a particular currents in the fine measurements
  //! if requested bin out of range, garbage -9999 is returned
  double FineCurrent(uint32_t i) const {
    if(i<_nfinemeasurements)
      return _finecurrent[i];
    return -9999.;
  }

  
  //! the number of temperature measurements
  uint32_t NTemps() const {
    return _ntemp;
  }
 /* 
  uint8_t SIPM_ID() const {
	  return _SIPM_ID;
  }
  
  uint8_t Module() const {
	  return _module;
  }
 */ 
  uint8_t Socket() const {
	  return _Socket;
  }
  
  uint64_t TimeStamp() const {
	  return _timestamp;
  }	  
  
   //! get the TrayID as a c++ string
  std::string TrayID() const {
    return std::string(_TrayID);
  }
  
     //! get the Sipm_ID_String as a c++ string
  std::string Sipm_ID_String() const {
    return std::string(_Sipm_ID_String);
  }
  
  
  //! get the array of initial temperatures
  double* InitialTemp() const {
    double *ret = new double[_ntemp];
    for(uint32_t i=0; i<_ntemp; i++){
      ret[i] = _temps_initial[i];
    }
    return ret;
  }
  
  
   //! get the array of final temperatures
  double* FinalTemp() const {
    double *ret = new double[_ntemp];
    for(uint32_t i=0; i<_ntemp; i++){
      ret[i] = _temps_final[i];
    }
    return ret;
  }

  //! dump the contents of the file onto stdout
  void Print() const
  {
    std::cout<<"Reading "<<(unsigned int)_nfinemeasurements<<" fine voltage scan"<<std::endl;
    for(uint32_t i=0; i<_nfinemeasurements; i++){
      std::cout<<"\t"<<_finevoltageSMU[i]<<" "<<_finevoltage[i]<<"  "<<_finecurrent[i]<<std::endl;
    }
   
    for(uint32_t i=0; i<_ntemp; i++){
      std::cout<<"Temperature: "<<_temps_initial[i]<<_temps_final[i]<<std::endl;
    }
  }
  
  double Idark0() const {
	  return _Idark0;
  }
  
  double Idark1() const {
	  return _Idark1;
  }
  
  double IdarkStartTemp() const {
	  return _IdarkStartTemp;
  }
  
  double forward_Resistance() const {
	  return _forward_Resistance;
  }
  
  double forward_Res_StartTemp() const {
	  return _forward_Res_StartTemp;
  }
  
  
  
/*
  void Write(const char* filename){
    _file = new std::fstream(filename,std::ios::out | std::ios::binary);
  //  WriteData<uint8_t>(_file,_header->Version());

    WriteData<uint32_t>(_file,_nfinemeasurements);
    for(uint32_t i=0; i<_nfinemeasurements; i++){
      WriteData<double>(_file,_finevoltage[i]);
      WriteData<double>(_file,_finecurrent[i]);
    }
    
    _file->close();
  }
*/
 private:
  std::fstream *_file;
  uint32_t _nfinemeasurements;
  double *_finevoltage;
  double *_finevoltageSMU;
  double *_finecurrent;
  uint32_t _ntemp;
  double *_temps_initial;
  double *_temps_final;
  uint8_t _TrayIDStringLength;
  char *_TrayID;
  uint8_t _SipmIDStringLength;
  char *_Sipm_ID_String;
  uint8_t _SIPM_ID;
  uint8_t _module;
  uint8_t _Socket;
  uint64_t _timestamp;
  double _DMM_resistance;
  //dark current measurement data
  double _Idark0;
  double _Idark1;
  double _Ileakage0;
  double _Ileakage1;
  double _IdarkStartTemp;
  double _IdarkEndTemp;
	
	//forward_resistance measurement data
  double _forward_Resistance;
  double _forward_Res_V1;
  double _forward_Res_V2;
  double _forward_Res_I1;
  double _forward_Res_I2;
  double _forward_Res_StartTemp;
  double _forward_Res_EndTemp;

};

#endif /* __IVBINARYFILE_H__ */
