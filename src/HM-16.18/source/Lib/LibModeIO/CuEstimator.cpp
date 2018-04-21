#include <iostream>

#include "CuEstimator.h"
#include "SessionHandler.h"

using namespace std;

#define TOTAL_DEPTH 4
#define NUMBER_OF_CU_TO_ESTIMATE (1 << ((TOTAL_DEPTH - 2) << 1))

#define RANGE_LOW_THRESHOLD 30

CuEstimator::CuEstimator()
{
  m_cSessionHandler = new SessionHandler();
}

CuEstimator::~CuEstimator()
{
  delete m_cSessionHandler;
  delete m_puiNumOfCu;
  delete[] m_ppCuMaxLuma;
  delete[] m_ppCuMinLuma;
  delete[] m_ppuhBestDepth;
}

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
  m_uiNumOfCtus = uiNumOfCtus;

  // m_puiNumOfCu = new UInt[5];
  // for (UInt depth = TOTAL_DEPTH - 1; depth >= 0; --depth)
  // {
  //   if (depth == TOTAL_DEPTH - 1)
  //   {
  //     m_puiNumOfCu[depth] = m_uiNumOfCtus >> 2;
  //     continue;
  //   }
  //   m_puiNumOfCu[depth] = m_puiNumOfCu[depth + 1] >> 2;
  // }

  m_ppCuMaxLuma = new Pel*[m_uiNumOfCtus];
  m_ppCuMinLuma = new Pel*[m_uiNumOfCtus];

  m_ppuhBestDepth = new UChar*[m_uiNumOfCtus];
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    m_ppuhBestDepth[uiCtuRsAddr] = new UChar[NUMBER_OF_CU_TO_ESTIMATE];
    memset(m_ppuhBestDepth[uiCtuRsAddr], TOTAL_DEPTH - 1, NUMBER_OF_CU_TO_ESTIMATE * sizeof(UChar));
    m_ppCuMaxLuma[uiCtuRsAddr] = new Pel[NUMBER_OF_CU_TO_ESTIMATE];
    memset(m_ppCuMaxLuma[uiCtuRsAddr], 0x0, NUMBER_OF_CU_TO_ESTIMATE * sizeof(Pel));    
    m_ppCuMinLuma[uiCtuRsAddr] = new Pel[NUMBER_OF_CU_TO_ESTIMATE];
    memset(m_ppCuMinLuma[uiCtuRsAddr], 0x7f, NUMBER_OF_CU_TO_ESTIMATE * sizeof(Pel));
  }
}

UChar **CuEstimator::estimateCtu(Pel **ppsCtusLuma)
{
  xProcessCtu(ppsCtusLuma);

  for (UChar depth = 0; depth <= TOTAL_DEPTH - 2; ++depth)
  {
    xSplitCuInDepth(ppsCtusLuma, depth);
  }

  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    for (UInt uiRsIdx = 0; uiRsIdx < 16; ++uiRsIdx)
    {
      UChar uhZIdx = (0x9 & uiRsIdx) | (0x4 & uiRsIdx << 1) | (0x2 & uiRsIdx >> 1);
      std::cout << (UInt)m_ppCuMaxLuma[uiCtuRsAddr][uhZIdx] << ' '
        << (UInt)m_ppCuMinLuma[uiCtuRsAddr][uhZIdx] << ' '
        << (UInt)m_ppuhBestDepth[uiCtuRsAddr][uhZIdx] << ' ';
      if (uiRsIdx % 4 == 3) std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  return m_ppuhBestDepth;
}

Void CuEstimator::xProcessCtu(Pel **ppsCtusLuma)
{
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    Pel *psCtuLuma = ppsCtusLuma[uiCtuRsAddr];

    for (UInt uiRsAddr = 0; uiRsAddr < m_uiMaxCuHeight * m_uiMaxCuWidth; ++uiRsAddr)
    {
      UChar uhZIdx = (0xc & uiRsAddr >> 8) | (0x3 & uiRsAddr >> 4);
      uhZIdx = (0x9 & uhZIdx) | (0x4 & uhZIdx << 1) | (0x2 & uhZIdx >> 1);

      m_ppCuMaxLuma[uiCtuRsAddr][uhZIdx] = max(psCtuLuma[uiRsAddr], m_ppCuMaxLuma[uiCtuRsAddr][uhZIdx]);
      m_ppCuMinLuma[uiCtuRsAddr][uhZIdx] = min(psCtuLuma[uiRsAddr], m_ppCuMinLuma[uiCtuRsAddr][uhZIdx]);
    }
  }
}

Void CuEstimator::xSplitCuInDepth(Pel **ppsCtusLuma, UChar uhDepth)
{
  if (uhDepth > TOTAL_DEPTH - 2) return;
  UChar step = 1 << ((TOTAL_DEPTH - 2 - uhDepth) << 1);
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    for (UChar uhZIdx = 0; uhZIdx < NUMBER_OF_CU_TO_ESTIMATE;  uhZIdx += step)
    {
      if (m_ppuhBestDepth[uiCtuRsAddr][uhZIdx] != TOTAL_DEPTH - 1) continue;
      Bool isSplit = true; 
      
      Pel maxLuma = (Pel)0, minLuma = (Pel)255;
      
      for (UChar i = uhZIdx; i < uhZIdx + step; ++i)
      {
        maxLuma = max(m_ppCuMaxLuma[uiCtuRsAddr][i], maxLuma);
        minLuma = min(m_ppCuMinLuma[uiCtuRsAddr][i], minLuma);
      }
      Pel range = maxLuma - minLuma;
      if (range > RANGE_LOW_THRESHOLD)
      {
        // To-Do
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
}
