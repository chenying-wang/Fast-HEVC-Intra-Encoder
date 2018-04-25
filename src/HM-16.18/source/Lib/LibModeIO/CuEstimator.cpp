#include <iostream>
#include <climits>
#include <memory>

#include "CuEstimator.h"
#include "SessionHandler.h"

/**
 * Constructor
*/
CuEstimator::CuEstimator()
{
  m_cSessionHandler = new SessionHandler();
}

/**
 * Destructor
*/
CuEstimator::~CuEstimator()
{
  delete m_cSessionHandler;

  for (UInt uiCuRsAddr = 0; uiCuRsAddr < m_uiNumOfCtus << ((TOTAL_DEPTH - 2) << 1); ++uiCuRsAddr)
  {
    delete[] m_ppsCusLuma[uiCuRsAddr];
  }
  delete[] m_ppsCusLuma;
  delete[] m_pusCuIdx;

  for (UInt i = 0; i < m_uiNumOfCtus; ++i)
  {
    delete[] m_ppCuMaxLuma[i];
    delete[] m_ppCuMinLuma[i];
    delete[] m_ppuhBestDepth[i];
  }
  delete[] m_ppCuMaxLuma;
  delete[] m_ppCuMinLuma;
  delete[] m_ppuhBestDepth;
}

/**
 * Initialize class member
*/
Void CuEstimator::init(const Int iPicWidth,
                       const Int iPicHeight,
                       const UInt uiMaxCuWidth,
                       const UInt uiMaxCuHeight,
                       const UInt uiNumOfCtus)
{
  m_iPicWidth = iPicWidth;
  m_iPicHeight = iPicHeight;
  m_uiMaxCuWidth = uiMaxCuWidth;
  m_uiMaxCuHeight = uiMaxCuHeight;
  m_uiMaxCuSize = m_uiMaxCuWidth * uiMaxCuHeight;
  m_uiNumOfCtus = uiNumOfCtus;

  m_ppCuMaxLuma = new Pel*[m_uiNumOfCtus];
  m_ppCuMinLuma = new Pel*[m_uiNumOfCtus];

  m_ppsCusLuma = new Pel*[m_uiNumOfCtus << ((TOTAL_DEPTH - 2) << 1)];
  
  for (UInt uiCuRsAddr = 0; uiCuRsAddr < m_uiNumOfCtus << ((TOTAL_DEPTH - 2) << 1); ++uiCuRsAddr)
  {
    m_ppsCusLuma[uiCuRsAddr] = new Pel[m_uiMaxCuSize];
  }
  m_pusCuIdx = new UShort[m_uiNumOfCtus << ((TOTAL_DEPTH - 2) << 1)];

  m_ppuhBestDepth = new UChar*[m_uiNumOfCtus];
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    m_ppuhBestDepth[uiCtuRsAddr] = new UChar[NUMBER_OF_CU_TO_ESTIMATE];
    m_ppCuMaxLuma[uiCtuRsAddr] = new Pel[NUMBER_OF_CU_TO_ESTIMATE];
    m_ppCuMinLuma[uiCtuRsAddr] = new Pel[NUMBER_OF_CU_TO_ESTIMATE];
  }
}

/**
 * 
*/
UChar **CuEstimator::estimateCtu(Pel **ppsCtusLuma)
{
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    memset(m_ppuhBestDepth[uiCtuRsAddr], TOTAL_DEPTH - 1, NUMBER_OF_CU_TO_ESTIMATE * sizeof(UChar));
    memset(m_ppCuMaxLuma[uiCtuRsAddr], 0x0, NUMBER_OF_CU_TO_ESTIMATE * sizeof(Pel));
    memset(m_ppCuMinLuma[uiCtuRsAddr], 0x7f, NUMBER_OF_CU_TO_ESTIMATE * sizeof(Pel));
  }

  xProcessCtu(ppsCtusLuma);
  for (UChar depth = 0; depth <= TOTAL_DEPTH - 2; ++depth)
  {
    xSplitCuInDepth(ppsCtusLuma, depth);
  }

  return m_ppuhBestDepth;
}

/**
 * 
*/
Void CuEstimator::xProcessCtu(Pel **ppsCtusLuma)
{
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    Pel *psCtuLuma = ppsCtusLuma[uiCtuRsAddr];

    for (UInt uiRsAddr = 0; uiRsAddr < m_uiMaxCuSize; ++uiRsAddr)
    {
      UChar uhZIdx = (0xc & uiRsAddr >> 8) | (0x3 & uiRsAddr >> 4);
      uhZIdx = (0x9 & uhZIdx) | (0x4 & uhZIdx << 1) | (0x2 & uhZIdx >> 1);

      m_ppCuMaxLuma[uiCtuRsAddr][uhZIdx] = std::max(psCtuLuma[uiRsAddr], m_ppCuMaxLuma[uiCtuRsAddr][uhZIdx]);
      m_ppCuMinLuma[uiCtuRsAddr][uhZIdx] = std::min(psCtuLuma[uiRsAddr], m_ppCuMinLuma[uiCtuRsAddr][uhZIdx]);
    }
  }
}

/**
 * 
*/
Void CuEstimator::xSplitCuInDepth(Pel **ppsCtusLuma, UChar uhDepth)
{
  if (uhDepth > TOTAL_DEPTH - 2) return;

  UInt uiCuWidth = 1 << (6 - uhDepth);
  UInt uiCuHeight = uiCuWidth;
  UChar step = 1 << ((TOTAL_DEPTH - 2 - uhDepth) << 1);

  UInt uiCuCount = 0;
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    for (UChar uhZIdx = 0; uhZIdx < NUMBER_OF_CU_TO_ESTIMATE;  uhZIdx += step)
    {
      if (m_ppuhBestDepth[uiCtuRsAddr][uhZIdx] != TOTAL_DEPTH - 1) continue;
      Bool isSplit = true; 
      
      Pel maxLuma = SHRT_MIN, minLuma = SHRT_MAX;
      
      for (UChar i = uhZIdx; i < uhZIdx + step; ++i)
      {
        maxLuma = std::max(m_ppCuMaxLuma[uiCtuRsAddr][i], maxLuma);
        minLuma = std::min(m_ppCuMinLuma[uiCtuRsAddr][i], minLuma);
      }
      Pel range = maxLuma - minLuma;
      if (range > RANGE_LOW_THRESHOLD)
      {
        for(UInt uiRow = 0; uiRow < uiCuHeight; ++uiRow)
        {
          UInt offset = uiRow << 6;
          memcpy(m_ppsCusLuma[uiCuCount] + (uiRow << (6 - uhDepth)), ppsCtusLuma[uiCtuRsAddr] + offset, uiCuWidth * sizeof(Pel));
        }
        m_pusCuIdx[uiCuCount] = (uiCtuRsAddr << ((TOTAL_DEPTH - 2) << 1)) | uhZIdx;
        ++uiCuCount;
      }
      else
      {
        isSplit = false;
      }

      if (!isSplit)
      {
        memset(m_ppuhBestDepth[uiCtuRsAddr] + uhZIdx, uhDepth, sizeof(UChar) * step);
      }
    }
  }

  Bool *pIsSplit = m_cSessionHandler->infer(m_ppsCusLuma, uiCuCount, uhDepth);
  for (UInt uiCuIdx = 0; uiCuIdx < uiCuCount; ++uiCuIdx)
  {
    if (!pIsSplit[uiCuIdx])
    {
      UInt uiCtuRsAddr = uiCuIdx >> ((TOTAL_DEPTH - 2) << 1);
      UChar uhZIdx = uiCuIdx & ((1 << ((TOTAL_DEPTH - 2) << 1)) - 1);
      memset(m_ppuhBestDepth[uiCtuRsAddr] + uhZIdx, uhDepth, sizeof(UChar) * step);
    }
  }
}
