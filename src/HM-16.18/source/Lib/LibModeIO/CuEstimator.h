#include <iostream>

#include "TLibCommon/CommonDef.h"

#include "SessionHandler.h"

#ifndef __CUESTIMATOR__
#define __CUESTIMATOR__

#define TOTAL_DEPTH 4
#define NUMBER_OF_CU_TO_ESTIMATE (1 << ((TOTAL_DEPTH - 2) << 1))

#define RANGE_LOW_THRESHOLD 30

class CuEstimator
{
private:
  SessionHandler *m_cSessionHandler;

  Int m_iPicWidth;
  Int m_iPicHeight;
  UInt m_uiMaxCuWidth;
  UInt m_uiMaxCuHeight;
  UInt m_uiMaxCuSize;
  UInt m_uiNumOfCtus;

  Pel **m_ppsCusLuma;
  UShort *m_pusCuIdx;

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
