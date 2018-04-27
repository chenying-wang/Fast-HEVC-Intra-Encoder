#include <iostream>

#include "TLibCommon/CommonDef.h"

#include "SessionWrapper.h"

#ifndef __CUESTIMATOR__
#define __CUESTIMATOR__

#define TOTAL_DEPTH 4
#define NUMBER_OF_CU_TO_ESTIMATE (1 << ((TOTAL_DEPTH - 2) << 1))

#define RANGE_LOW_THRESHOLD 30

class CuEstimator
{
private:
  SessionWrapper *m_cSessionWrapper;

  Int m_iPicWidth;
  Int m_iPicHeight;
  UInt m_uiLogMaxCuWidth;
  UInt m_uiLogMaxCuHeight;
  UInt m_uiMaxCuSize;
  UInt m_uiFrameWidthInCtus;
  UInt m_uiFrameHeightInCtus;
  UInt m_uiNumOfCtus;

  Pel **m_ppsCusLuma;
  UShort *m_pusCuIdx;

  Pel **m_ppsCuMaxLuma;
  Pel **m_ppsCuMinLuma;
  UChar **m_ppuhBestDepth;

protected:
  Void xProcessCtu(Pel **ppsCtusLuma);
  Void xSplitCuInDepth(Pel **ppsCtusLuma, UChar uhDepth);

public:
  CuEstimator();
  virtual ~CuEstimator();
  Void init(const Int iPicWidth, const Int iPicHeight, const UInt uiLogMaxCuWidth, const UInt uiLogMaxCuHeight, const UInt uiFrameWidthInCtus, const UInt uiFrameHeightInCtus, const UInt uiNumOfCtus);
  UChar **estimateCtus(Pel **ppsCtusLuma);
};

#endif // __CUESTIMATOR__
