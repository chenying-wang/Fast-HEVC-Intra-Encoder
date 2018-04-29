#include <iostream>
#include <fstream>
#include <string>
#include <regex>

#include "CuModeIO.h"
#include "CuEstimator.h"

/**
 * Constructor
*/
CuModeIO::CuModeIO(IOMode mode) :
  m_filename(string("default").append(_CU_MODE_OUTPUT_FILENAME_EXTENSION)),
  m_iEncodedPictures(0)
{
  m_mode = mode;
  m_cCuEstimator = new CuEstimator();
}

/**
 * Destructor
*/
CuModeIO::~CuModeIO()
{
  delete m_cCuEstimator;

  if (m_mode == IN)
  {
    for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
    {
      delete[] m_ppsCtuLuma[uiCtuRsAddr];
    }
    delete[] m_ppsCtuLuma;
  }
  else if (m_mode == OUT)
  {
    delete[] m_psPicCuMode->psCuMode;
  }

  delete[] m_psPicCuMode;
}

/**
 * Initialize class member
*/
Void CuModeIO::init(const std::string &filename,
                    const Int iNumPictures,
                    const Int iPicWidth,
                    const Int iPicHeight,
                    const UInt uiMaxCuWidth,
                    const UInt uiMaxCuHeight,
                    const UInt uiMaxTotalCUDepth)
{
  assert(m_mode == IN || m_mode == OUT);
  if (m_mode == OUT) xSetFilename(filename, uiMaxCuWidth);

  m_iNumPictures = iNumPictures;
  m_iPicWidth = iPicWidth;
  m_iPicHeight = iPicHeight;


  m_uiMaxCuWidth = uiMaxCuWidth;
  for (m_uiLogMaxCuWidth = 1; m_uiMaxCuWidth >> m_uiLogMaxCuWidth > 1; ++m_uiLogMaxCuWidth) ;
  m_uiMaxCuHeight = uiMaxCuHeight;
  for (m_uiLogMaxCuHeight = 1; m_uiMaxCuHeight >> m_uiLogMaxCuHeight > 1; ++m_uiLogMaxCuHeight) ;

  m_uiMaxTotalCuDepth = uiMaxTotalCUDepth;

  m_uiFrameWidthInCtus = (m_iPicWidth >> m_uiLogMaxCuWidth) + (m_iPicWidth % m_uiMaxCuWidth ? 1 :0 );
  m_uiFrameHeightInCtus = (m_iPicHeight >> m_uiLogMaxCuHeight) + (m_iPicHeight % m_uiMaxCuHeight ? 1 : 0);
  m_uiNumOfCtus = m_uiFrameWidthInCtus * m_uiFrameHeightInCtus;
  m_uiNumPartInCtuWidth = 1 << m_uiMaxTotalCuDepth;
  m_uiNumPartInCtuHeight = 1 << m_uiMaxTotalCuDepth;

  m_psPicCuMode = new PicCuMode();

  if (m_mode == IN)
  {
    m_ppsCtuLuma = new Pel*[m_uiNumOfCtus];
    for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
    {
      m_ppsCtuLuma[uiCtuRsAddr] = new Pel[1 << (m_uiLogMaxCuWidth + m_uiLogMaxCuHeight)];
    }
    m_cCuEstimator->init(m_iPicWidth, m_iPicHeight, m_uiLogMaxCuWidth, m_uiLogMaxCuHeight, m_uiFrameWidthInCtus, m_uiFrameHeightInCtus, m_uiNumOfCtus);
  }
  else if (m_mode == OUT)
  {
    m_file.open(m_filename, fstream::out);
    m_psPicCuMode->psCuMode = new CuMode[m_uiNumOfCtus];
  }
}

/**
 * Read all CU Mode of a picture
*/
UChar **CuModeIO::read(TComPic *&pcPic)
{
  if (pcPic->getSlice(0)->getSliceType() != I_SLICE)
  {
    std::cerr << "WARNING: POC: " << pcPic->getPOC() << " is not I_SLICE!";
    return NULL;
  }

  m_psPicCuMode->uiPOC = pcPic->getPOC();
  TComPicYuv *pcPicYuv = pcPic->getPicYuvTrueOrg();

  const UInt stride = pcPicYuv->getStride(COMPONENT_Y);

  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    Pel *ctuLumaAddr = pcPicYuv->getAddr(COMPONENT_Y, uiCtuRsAddr);
    for (UInt uiRow = 0; uiRow < m_uiMaxCuHeight; ++uiRow)
    {
      UInt offset = uiRow * stride;
      memcpy(m_ppsCtuLuma[uiCtuRsAddr] + (uiRow << m_uiLogMaxCuWidth), ctuLumaAddr + offset, sizeof(Pel) << m_uiLogMaxCuWidth);
    }
  }

  return m_cCuEstimator->estimateCtus(m_ppsCtuLuma);
}

/**
 * Write POC and every CU Mode of a I_SLICE picture
*/
Void CuModeIO::write(TComPic *&pcPic)
{
  if (pcPic->getSlice(0)->getSliceType() != I_SLICE)
  {
    std::cerr << "WARNING: POC: " << pcPic->getPOC() << " is not I_SLICE!";
    return;
  }

  m_psPicCuMode->uiPOC = pcPic->getPOC();
  xExtractMode(pcPic);
  
  TComPicYuv *pcPicYuv = pcPic->getPicYuvTrueOrg();

  const UInt stride = pcPicYuv->getStride(COMPONENT_Y);
  const CuMode *cuMode = m_psPicCuMode->psCuMode;

  // for every ctu in frame
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiNumOfCtus; ++uiCtuRsAddr)
  {
    Pel *pPelLuma = pcPicYuv->getAddr(COMPONENT_Y, uiCtuRsAddr);

    // write luma in ctu
    for (UInt uiRow = 0; uiRow < m_uiMaxCuHeight; ++uiRow)
    {
      UInt offset = uiRow * stride;
      for (UInt uiCol = 0; uiCol < m_uiMaxCuWidth; ++uiCol)
      {
        m_file << pPelLuma[offset + uiCol] << ',';
      }
    }

    // write whether ctu split or not
    m_file << (UInt)cuMode[uiCtuRsAddr].puhDepth[0];

    // if ((UInt)cuMode[uiCtuRsAddr].puhDepth[0] == 0) continue;
    
    // for every partition in ctu
    // for (UInt uiRsIdx = 0; uiRsIdx < m_uiNumPartInCtuWidth * m_uiNumPartInCtuHeight; ++uiRsIdx)
    // {
    //   UInt uiZIdx = g_auiRasterToZscan[uiRsIdx];
    //   std::cout << std::setw(2) << (UInt)cuMode[uiCtuRsAddr].puhDepth[uiZIdx] << ' '
    //     << std::setw(2) << (UInt)cuMode[uiCtuRsAddr].puhLumaIntraDir[uiZIdx] << ' '
    //     << (cuMode[uiCtuRsAddr].pePartSize[0]==SIZE_2Nx2N ? "2N":"1N");
    //   if ((uiRsIdx + 1) % m_uiNumPartInCtuWidth) std::cout << ' ';
    //   else std::cout << std::endl;
    // }

    std::cout << std::endl;
  }
  ++m_iEncodedPictures;
}

/**
 * Set output filename in accord with yuv filename
*/
Void CuModeIO::xSetFilename(const std::string &filename, UInt maxCUWidth)
{
  std::string temp_filename = filename;
  std::smatch match;
  std::string pattern("/([0-9A-Za-z_]+)\\.yuv$");

  std::regex_search(temp_filename, match, std::regex(pattern));
  if (!match[1].matched) return;
  m_filename = match[1].str() + "_" + std::to_string(maxCUWidth)
    + _CU_MODE_OUTPUT_FILENAME_EXTENSION;
  std::cout << std::setw(48) << "Cu Mode IO File : " << m_filename << std::endl << std::endl;
}

/**
 * Extract each CU mode in raster order from a given picture
*/ 
Void CuModeIO::xExtractMode(TComPic *&pcPic)
{
  CuMode *cuMode = m_psPicCuMode->psCuMode;
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiFrameWidthInCtus * m_uiFrameHeightInCtus; ++uiCtuRsAddr)
  {
    TComDataCU* pCtu = pcPic->getCtu(uiCtuRsAddr);
    if (m_mode == IN) continue;
    cuMode[uiCtuRsAddr].puhDepth = pCtu->getDepth();
    cuMode[uiCtuRsAddr].puhLumaIntraDir = pCtu->getIntraDir(CHANNEL_TYPE_LUMA);
    cuMode[uiCtuRsAddr].pePartSize = pCtu->getPartitionSize();
  }
}
