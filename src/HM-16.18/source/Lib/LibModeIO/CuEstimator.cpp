#include <iostream>
#include <climits>
#include <cstring>

#include "CuEstimator.h"
#include "SessionWrapper.h"

/**
 * Constructor
*/
CuEstimator::CuEstimator()
{
  m_pcSessionWrapper = new SessionWrapper();
}

/**
 * Destructor
*/
CuEstimator::~CuEstimator()
{
  delete m_pcSessionWrapper;

  UInt uiMaxNumOfCus = m_uiNumOfCtus << ((m_uiMaxTotalCuDepth - 1) << 1);
  delete[] m_pusCuIdx;
  for (UInt uiCuRsAddr = 0; uiCuRsAddr < uiMaxNumOfCus; ++uiCuRsAddr)
  {
    delete[] m_ppsCusLuma[uiCuRsAddr];
  }
  delete[] m_ppsCusLuma;
  
  // for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  // {
  //   delete[] m_ppsCuMaxLuma[uiCtuRsAddr];
  //   delete[] m_ppsCuMinLuma[uiCtuRsAddr];
  //   delete[] m_ppuhBestDepth[uiCtuRsAddr];
  // }
  delete[] m_ppsCuMaxLuma;
  delete[] m_ppsCuMinLuma;
  delete[] m_ppuhBestDepth;

  delete[] m_puiCuWidth;
  delete[] m_puiCuHeight;
  delete[] m_puhStep;
  delete[] m_puhBestDepthSize;

  delete[] m_puhRsAddrToRsIdx;
  delete[] m_puhRsIdxToZIdx;
}

/**
 * Initialize class member
*/
Void CuEstimator::init(const Int iPicWidth,
                       const Int iPicHeight,
                       const UInt uiLogMaxCuWidth,
                       const UInt uiLogMaxCuHeight,
                       const UInt uiMaxTotalCuDepth,
                       const UInt uiFrameWidthInCtus,
                       const UInt uiFrameHeightInCtus,
                       const UInt uiNumOfCtus)
{
  m_iPicWidth = iPicWidth;
  m_iPicHeight = iPicHeight;
  m_uiLogMaxCuWidth = uiLogMaxCuWidth;
  m_uiLogMaxCuHeight = uiLogMaxCuHeight;
  m_uiMaxTotalCuDepth = uiMaxTotalCuDepth;
  
  m_uiMaxCuSize = 1 << (m_uiLogMaxCuWidth + m_uiLogMaxCuHeight);
  m_uiFrameWidthInCtus = uiFrameWidthInCtus;
  m_uiFrameHeightInCtus = uiFrameHeightInCtus;
  m_uiNumOfCtus = uiNumOfCtus;
  m_uhNumOfCuToEst = 1 << ((m_uiMaxTotalCuDepth - 1) << 1);

  UInt uiMaxNumOfCus = m_uiNumOfCtus << ((m_uiMaxTotalCuDepth - 1) << 1);
  m_pusCuIdx = new UShort[uiMaxNumOfCus];
  m_ppsCusLuma = new Pel*[uiMaxNumOfCus];
  for (UInt uiCuRsAddr = 0; uiCuRsAddr < uiMaxNumOfCus; ++uiCuRsAddr)
  {
    m_ppsCusLuma[uiCuRsAddr] = new Pel[m_uiMaxCuSize];
  }

  m_ppsCuMaxLuma = new Pel*[m_uiNumOfCtus];
  m_ppsCuMinLuma = new Pel*[m_uiNumOfCtus];
  m_ppuhBestDepth = new UChar*[m_uiNumOfCtus];
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    m_ppsCuMaxLuma[uiCtuRsAddr] = new Pel[m_uhNumOfCuToEst];
    m_ppsCuMinLuma[uiCtuRsAddr] = new Pel[m_uhNumOfCuToEst];
    m_ppuhBestDepth[uiCtuRsAddr] = new UChar[m_uhNumOfCuToEst];
  }

  m_puiCuWidth = new UInt[m_uiMaxTotalCuDepth];
  m_puiCuHeight = new UInt[m_uiMaxTotalCuDepth];
  m_puhStep = new UChar[m_uiMaxTotalCuDepth];
  m_puhBestDepthSize = new UChar[m_uiMaxTotalCuDepth];
  for (UShort uhDepth = 0; uhDepth < m_uiMaxTotalCuDepth; ++uhDepth)
  {
    m_puiCuWidth[uhDepth] = 1 << (m_uiLogMaxCuWidth - uhDepth);
    m_puiCuHeight[uhDepth] = 1 << (m_uiLogMaxCuHeight - uhDepth);
    m_puhStep[uhDepth] = 1 << ((m_uiMaxTotalCuDepth - 1 - uhDepth) << 1);
    m_puhBestDepthSize[uhDepth] = sizeof(UChar) << ((m_uiMaxTotalCuDepth - 1 - uhDepth) << 1);
  }

  m_puhRsAddrToRsIdx = new UChar[m_uiMaxCuSize];
  for (UInt uiRsAddr = 0; uiRsAddr < m_uiMaxCuSize; ++uiRsAddr)
  {
    m_puhRsAddrToRsIdx[uiRsAddr] = (0xc & uiRsAddr >> 8) | (0x3 & uiRsAddr >> 4);
  }
  m_puhRsIdxToZIdx = new UChar[m_uhNumOfCuToEst];
  m_puhZIdxToRsIdx = m_puhRsIdxToZIdx;
  for (UChar uhRsIdx = 0; uhRsIdx < m_uhNumOfCuToEst;  ++uhRsIdx)
  {
    m_puhRsIdxToZIdx[uhRsIdx] = (0x9 & uhRsIdx) | (0x4 & uhRsIdx << 1) | (0x2 & uhRsIdx >> 1);
  }

  m_puiSum = new UInt[m_uiMaxTotalCuDepth + 1];

  m_pcSessionWrapper->init(uiMaxNumOfCus, m_uiLogMaxCuWidth, m_uiMaxTotalCuDepth);
}

/**
 * Estimate best depth of each CTUs in a picture
*/
UChar **CuEstimator::estimateCtus(Pel **ppsCtusLuma)
{
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    memset(m_ppsCuMaxLuma[uiCtuRsAddr], 0x0, m_uhNumOfCuToEst * sizeof(Pel));
    memset(m_ppsCuMinLuma[uiCtuRsAddr], 0x7f, m_uhNumOfCuToEst * sizeof(Pel));
    memset(m_ppuhBestDepth[uiCtuRsAddr], m_uiMaxTotalCuDepth, m_uhNumOfCuToEst * sizeof(UChar));
  }

  xProcessCtu(ppsCtusLuma);
  
  for (UChar depth = 0; depth <= m_uiMaxTotalCuDepth - 1; ++depth)
  {
    xSplitCuInDepth(ppsCtusLuma, depth);
  }

  for (UChar depth = 0; depth <= m_uiMaxTotalCuDepth; ++depth)
  {
    m_puiSum[depth] = 0;
  }
  
  // for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  // {
  //   for (UChar uhZIdx = 0; uhZIdx < m_uhNumOfCuToEst;  ++uhZIdx)
  //   {
  //     if (m_ppuhBestDepth[uiCtuRsAddr][uhZIdx] > 3) std::cout << m_ppuhBestDepth[uiCtuRsAddr][uhZIdx] << std::endl;
  //     ++m_puiSum[m_ppuhBestDepth[uiCtuRsAddr][uhZIdx]];
  //   }
  // }

  // for (UChar depth = 0; depth <= m_uiMaxTotalCuDepth; ++depth)
  // {
  //   std::cout << "Depth " << (UInt)depth << "    " << m_puiSum[depth] << std::endl;
  // }

  return m_ppuhBestDepth;
}

/**
 * Calculate max/min luma of each CU
*/
Void CuEstimator::xProcessCtu(Pel **ppsCtusLuma)
{
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    Pel *psCtuLuma = ppsCtusLuma[uiCtuRsAddr];

    for (UInt uiRsAddr = 0; uiRsAddr < m_uiMaxCuSize; ++uiRsAddr)
    {
      UChar uhZIdx = m_puhRsIdxToZIdx[m_puhRsAddrToRsIdx[uiRsAddr]];
      m_ppsCuMaxLuma[uiCtuRsAddr][uhZIdx] = std::max(psCtuLuma[uiRsAddr], m_ppsCuMaxLuma[uiCtuRsAddr][uhZIdx]);
      m_ppsCuMinLuma[uiCtuRsAddr][uhZIdx] = std::min(psCtuLuma[uiRsAddr], m_ppsCuMinLuma[uiCtuRsAddr][uhZIdx]);
    }
  }
}

/**
 * Decide if split all CTUs/CUs in a particular depth
*/
Void CuEstimator::xSplitCuInDepth(Pel **ppsCtusLuma, UChar uhDepth)
{
  UInt uiCuWidth = m_puiCuWidth[uhDepth];
  UInt uiCuHeight = m_puiCuHeight[uhDepth];
  UChar step = m_puhStep[uhDepth];
  UChar uhBestDepthSize = m_puhBestDepthSize[uhDepth];

  UInt uiCuCount = 0;
  for (UChar uhZIdx = 0; uhZIdx < m_uhNumOfCuToEst;  uhZIdx += step)
  {
    UChar uhRsIdx = m_puhZIdxToRsIdx[uhZIdx];
    UInt uiCuOffsetX = (uhRsIdx & 0x3) << (m_uiLogMaxCuWidth - (m_uiMaxTotalCuDepth - 1));
    UInt uiCuOffsetY = (uhRsIdx >> 2) << (m_uiLogMaxCuHeight - (m_uiMaxTotalCuDepth - 1));

    for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
    {
      UInt uiCtuPelX = (uiCtuRsAddr % m_uiFrameWidthInCtus) << m_uiLogMaxCuWidth;
      UInt uiCtuPelY = (uiCtuRsAddr / m_uiFrameHeightInCtus) << m_uiLogMaxCuHeight;
      if (uiCtuPelX + uiCuOffsetX > m_iPicWidth || uiCtuPelY + uiCuOffsetY > m_iPicHeight) continue;
      if (m_ppuhBestDepth[uiCtuRsAddr][uhZIdx] != m_uiMaxTotalCuDepth) continue; 
      
      Pel maxLuma = SHRT_MIN, minLuma = SHRT_MAX;
      for (UChar j = 0; j < step; ++j)
      {
        UChar i = uhZIdx | j;
        maxLuma = std::max(m_ppsCuMaxLuma[uiCtuRsAddr][i], maxLuma);
        minLuma = std::min(m_ppsCuMinLuma[uiCtuRsAddr][i], minLuma);
      }
      Pel range = maxLuma - minLuma;
      if (range > RANGE_LOW_THRESHOLD)
      {
        for(UInt uiRow = 0; uiRow < uiCuHeight; ++uiRow)
        {
          UInt offset = uiRow << m_uiLogMaxCuWidth;
          memcpy(m_ppsCusLuma[uiCuCount] + (uiRow << (m_uiLogMaxCuWidth - uhDepth)), ppsCtusLuma[uiCtuRsAddr] + offset, uiCuWidth * sizeof(Pel));
        }
        m_pusCuIdx[uiCuCount] = (uiCtuRsAddr << ((m_uiMaxTotalCuDepth - 1) << 1)) | uhZIdx;
        ++uiCuCount;
      }
      else
      {
        memset(m_ppuhBestDepth[uiCtuRsAddr] + uhZIdx, uhDepth, uhBestDepthSize);
      }
    }
  }

  Bool *pIsSplit = m_pcSessionWrapper->infer(m_ppsCusLuma, uiCuCount, uhDepth);
  for (UInt uiCuIdx = 0; uiCuIdx < uiCuCount; ++uiCuIdx)
  {
    if (!pIsSplit[uiCuIdx])
    {
      UInt uiCtuRsAddr = m_pusCuIdx[uiCuCount] >> ((m_uiMaxTotalCuDepth - 1) << 1);
      UChar uhZIdx = m_pusCuIdx[uiCuCount] & ((1 << ((m_uiMaxTotalCuDepth - 1) << 1)) - 1);
      memset(m_ppuhBestDepth[uiCtuRsAddr] + uhZIdx, uhDepth, uhBestDepthSize);
    }
  }
}
