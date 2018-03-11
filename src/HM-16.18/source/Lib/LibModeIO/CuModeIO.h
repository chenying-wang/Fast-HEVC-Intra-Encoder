#include <iostream>
#include <fstream>

#include "TLibCommon/CommonDef.h"
#include "TLibCommon/TComSlice.h"
#include "TLibCommon/TComPic.h"

#ifndef __CUMODEIO__
#define __CUMODEIO__

#define _DISABLE_SIZE_NXN                              0
#define _CU_MODE_OUTPUT_FILENAME_EXTENSION             ".csv"

enum IOMode
{
  IN = 0,
  OUT = 1
};

class CuModeIO
{
private:
  fstream m_file;
  std::string m_filename;
  IOMode m_mode;
  Int m_iEncodedPictures;

  Int m_iNumPictures;
  Int m_iPicWidth;
  Int m_iPicHeight;
  UInt m_uiMaxCuWidth;
  UInt m_uiMaxCuHeight;
  UInt m_uiMaxTotalCuDepth;

  UInt m_uiFrameWidthInCtus;
  UInt m_uiFrameHeightInCtus;
  UInt m_uiNumPartInCtuWidth;
  UInt m_uiNumPartInCtuHeight;

  struct CuMode
  {
    UChar* puhDepth;
    UChar* puhLumaIntraDir;
    SChar* pePartSize;
  };

  struct PicCuMode
  {
    UInt uiPOC;
    CuMode* psCuMode;
  }* m_psPicCuMode;

protected:
  Void xSetFilename(const std::string &filename, UInt maxCUWidth);
  Void xExtractMode(TComPic*& pcPic);

public:
  CuModeIO(IOMode mode);
  virtual ~CuModeIO();
  Void init(const std::string &filename, const Int numPictures, const Int picWidth, const Int picHeight, const UInt maxCUWidth, const UInt maxCUHeight, const UInt maxTotalCUDepth);
  Void read();
  Void write(TComPic* &pcPic);
};

#endif // __CUMODEIO__