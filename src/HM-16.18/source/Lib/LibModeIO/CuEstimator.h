#include <iostream>

#include "TLibCommon/CommonDef.h"

#include "SessionWrapper.h"

#ifndef __CUESTIMATOR__
#define __CUESTIMATOR__

class CuEstimator
{
private:
  SessionWrapper *m_pcSessionWrapper;

  Int m_iPicWidth;
  Int m_iPicHeight;
  UInt m_uiLogMaxCuWidth;
  UInt m_uiLogMaxCuHeight;
  UInt m_uiMaxTotalCuDepth;
  Int m_iQP;
  
  UInt m_uiMaxCuSize;
  UInt m_uiFrameWidthInCtus;
  UInt m_uiFrameHeightInCtus;
  UInt m_uiNumOfCtus;
  UChar m_uhNumOfCuToEst;

  Pel **m_ppsCusLuma;
  UShort *m_pusCuIdx;

  Pel **m_ppsCuMaxLuma;
  Pel **m_ppsCuMinLuma;
  UChar **m_ppuhBestDepth;

  UInt *m_puiCuWidth;
  UInt *m_puiCuHeight;
  UChar *m_puhStep;
  UChar *m_puhBestDepthSize;

  UChar *m_puhRsAddrToRsIdx;
  UChar *m_puhRsIdxToZIdx;
  UChar *m_puhZIdxToRsIdx;

  UInt *m_puiSum;

  const UInt *m_uiRangeLowThreshold = new UInt[3]{20, 25, 30};

protected:
  Void xProcessCtu(Pel **ppsCtusLuma);
  Void xSplitCuInDepth(Pel **ppsCtusLuma, UChar uhDepth);

public:
  CuEstimator();
  virtual ~CuEstimator();
  Void init(const Int iPicWidth, const Int iPicHeight, const UInt uiLogMaxCuWidth, const UInt uiLogMaxCuHeight, const UInt uiMaxTotalCuDepth, const UInt uiFrameWidthInCtus, const UInt uiFrameHeightInCtus, const UInt uiNumOfCtus, const Int iQp);
  UChar **estimateCtus(Pel **ppsCtusLuma);
};

#endif // __CUESTIMATOR__
