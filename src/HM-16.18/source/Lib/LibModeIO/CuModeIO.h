#include <iostream>
#include <fstream>

#include "TLibCommon/CommonDef.h"
#include "TLibCommon/TComPic.h"

#include "CuEstimator.h"

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
  CuEstimator *m_pcCuEstimator;
  
  fstream m_file;
  std::string m_filename;
  IOMode m_mode;
  Int m_iEncodedPictures;

  Int m_iNumPictures;
  Int m_iPicWidth;
  Int m_iPicHeight;
  UInt m_uiMaxCuWidth;
  UInt m_uiLogMaxCuWidth;
  UInt m_uiMaxCuHeight;
  UInt m_uiLogMaxCuHeight;
  UInt m_uiMaxTotalCuDepth;

  UInt m_uiFrameWidthInCtus;
  UInt m_uiFrameHeightInCtus;
  UInt m_uiNumOfCtus;
  UInt m_uiNumPartInCtuWidth;
  UInt m_uiNumPartInCtuHeight;

  Pel **m_ppsCtuLuma;

  struct CuMode
  {
    UChar *puhDepth;
    UChar *puhLumaIntraDir;
    SChar *pePartSize;
  };

  struct PicCuMode
  {
    UInt uiPOC;
    CuMode *psCuMode;
  } *m_psPicCuMode;

protected:
  Void xSetFilename(const std::string &filename, UInt maxCUWidth);
  Void xExtractMode(TComPic *&pcPic);

public:
  CuModeIO(IOMode mode);
  virtual ~CuModeIO();
  Void init(const std::string &filename, const Int iNumPictures, const Int iPicWidth, const Int iPicHeight, const UInt uiMaxCuWidth, const UInt uiMaxCuHeight, const UInt uiMaxTotalCUDepth);
  UChar **read(TComPic *&pcPic);
  Void write(TComPic *&pcPic);
};

#endif // __CUMODEIO__
