#include <iostream>
#include <fstream>
#include <string>
#include <regex>

#include "CuModeIO.h"

/**
 * Constructor
*/
CuModeIO::CuModeIO(IOMode mode) :
  m_filename(string("default").append(_CU_MODE_OUTPUT_FILENAME_EXTENSION)),
  m_iEncodedPictures(0)
{
  m_mode = mode;
}

/**
 * Destructor
*/
CuModeIO::~CuModeIO()
{
  delete[] m_psPicCuMode->psCuMode;
  delete m_psPicCuMode;
}

/**
 * Initialize class member
*/
Void CuModeIO::init(const std::string &filename,
                    const Int numPictures,
                    const Int picWidth,
                    const Int picHeight,
                    const UInt maxCUWidth,
                    const UInt maxCUHeight,
                    const UInt maxTotalCUDepth)
{
  assert(m_mode == IN || m_mode == OUT);
  xSetFilename(filename, maxCUWidth);
  if (m_mode == IN)
  {
    m_file.open(m_filename, fstream::in);
  }
  else if (m_mode == OUT)
  {
    m_file.open(m_filename, fstream::out);
  }

  m_iNumPictures = numPictures;
  m_iPicWidth = picWidth;
  m_iPicHeight = picHeight;
  m_uiMaxCuWidth = maxCUWidth;
  m_uiMaxCuHeight = maxCUHeight;
  m_uiMaxTotalCuDepth = maxTotalCUDepth;

  m_uiFrameWidthInCtus = m_iPicWidth / m_uiMaxCuWidth + (m_iPicWidth % m_uiMaxCuWidth ? 1 :0 );
  m_uiFrameHeightInCtus = m_iPicHeight / m_uiMaxCuHeight + (m_iPicHeight % m_uiMaxCuHeight ? 1 : 0);
  m_uiNumPartInCtuWidth = 1 << m_uiMaxTotalCuDepth;
  m_uiNumPartInCtuHeight = 1 << m_uiMaxTotalCuDepth;

  m_psPicCuMode = new PicCuMode();
  m_psPicCuMode->psCuMode = new CuMode[m_uiFrameWidthInCtus * m_uiFrameHeightInCtus];  
}

/**
 * TODO: Read all CU Mode of a picture
*/
Void CuModeIO::read()
{

}

/**
 * Write POC and every CU Mode of a I_SLICE picture
*/
Void CuModeIO::write(TComPic* &pcPic)
{
  if (pcPic->getSlice(0)->getSliceType() != I_SLICE)
  {
    std::cerr << "WARNING: POC: " << pcPic->getPOC() << " is not I_SLICE!";
    return;
  }

  xExtractMode(pcPic);
  
  TComPicYuv* pcPicYuv = pcPic->getPicYuvTrueOrg();\

  const UInt stride = pcPicYuv->getStride(COMPONENT_Y);
  const CuMode* cuMode = m_psPicCuMode->psCuMode;

  // for every ctu in frame
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiFrameWidthInCtus * m_uiFrameHeightInCtus; ++uiCtuRsAddr)
  {
    // m_file << "CTU Raster Idx: " << uiCtuRsAddr << std::endl;
    Pel* pPelLuma = pcPicYuv->getAddr(COMPONENT_Y, uiCtuRsAddr);

    // write luma in ctu
    for(UInt uiRow = 0; uiRow < m_uiMaxCuHeight; ++uiRow)
    {
      UInt offset = uiRow * stride;
      for(UInt uiCol = 0; uiCol < m_uiMaxCuWidth; ++uiCol)
      {
        m_file << pPelLuma[offset + uiCol] << ',';
      }
    }

    // write whether ctu split or not
    m_file << (UInt)cuMode[uiCtuRsAddr].puhDepth[0];

    // for every partition in ctu
    // for (UInt uiRsIdx = 0; uiRsIdx < m_uiNumPartInCtuWidth * m_uiNumPartInCtuHeight; ++uiRsIdx)
    // {
    //   UInt uiZIdx = g_auiRasterToZscan[uiRsIdx];

    //   m_file << std::setw(2) << (UInt)cuMode[uiCtuRsAddr].puhDepth[uiZIdx] << ' '
    //     << std::setw(2) << (UInt)cuMode[uiCtuRsAddr].puhLumaIntraDir[uiZIdx] << ' '
    //     << (cuMode[uiCtuRsAddr].pePartSize[0]==SIZE_2Nx2N ? "2N":"1N");
    //   if ((uiRsIdx + 1) % m_uiNumPartInCtuWidth) m_file << ' ';
    //   else m_file << std::endl;
    // }

    m_file << std::endl;
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
Void CuModeIO::xExtractMode(TComPic* &pcPic)
{
  m_psPicCuMode->uiPOC = pcPic->getPOC();
  CuMode* cuMode = m_psPicCuMode->psCuMode;
  for (UInt uiCtuRsAddr = 0; uiCtuRsAddr < m_uiFrameWidthInCtus * m_uiFrameHeightInCtus; ++uiCtuRsAddr)
  {
    TComDataCU* pCtu = pcPic->getCtu(uiCtuRsAddr);
    cuMode[uiCtuRsAddr].puhDepth = pCtu->getDepth();
    cuMode[uiCtuRsAddr].puhLumaIntraDir = pCtu->getIntraDir(CHANNEL_TYPE_LUMA);
    cuMode[uiCtuRsAddr].pePartSize = pCtu->getPartitionSize();
  }
}