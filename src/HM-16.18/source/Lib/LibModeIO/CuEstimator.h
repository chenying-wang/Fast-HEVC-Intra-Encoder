#include <iostream>

#include "TLibCommon/CommonDef.h"

#include "SessionHandler.h"

using namespace std;

#ifndef __CUESTIMATOR__
#define __CUESTIMATOR__

class CuEstimator
{
private:
  SessionHandler *m_cSessionHandler;

  Int m_iPicWidth;
  Int m_iPicHeight;
  UInt m_uiMaxCuWidth;
  UInt m_uiMaxCuHeight;
  UInt m_uiNumOfCtus;
  UInt *m_puiNumOfCu;

  Pel **m_ppCuMaxLuma;
  Pel **m_ppCuMinLuma;

  UChar **m_ppuhBestDepth;

protected:
  Void xProcessCtu(Pel **ppsCtusLuma);
  Void xSplitCuInDepth(Pel **ppsCtusLuma, UChar uhDepth);

public:
  CuEstimator();
  virtual ~CuEstimator();
  Void init(const Int iPicWidth, const Int iPicHeight, const UInt uiMaxCuWidth, const UInt uiMaxCuHeight, const UInt uiNumOfCtus);
  UChar **estimateCtu(Pel **ppsCtusLuma);
};

#endif // __CUESTIMATOR__
