#include <iostream>

#include "SessionHandler.h"
// #include "tensorflow/core/platform/env.h"
// #include "tensorflow/core/public/session.h"

// using namespace tensorflow;
// using namespace tensorflow::ops;

/**
 * Constructor
*/
SessionHandler::SessionHandler()
{

}

/**
 * Destructor
*/
SessionHandler::~SessionHandler()
{

}

/**
 * 
*/
Bool *SessionHandler::infer(Pel **ppsCusLuma, UInt uiNumOfCus, UChar uhDepth)
{
  Bool *pIsSplit = new Bool[uiNumOfCus];
  UInt uiCuSize = 1 << ((6 - uhDepth) << 1);

  for (UInt uiCuIdx = 0; uiCuIdx < uiNumOfCus; ++uiCuIdx)
  {
    std::cout << "uiCuIdx..." << uiCuIdx << std::endl;
    for (UInt i = 0; i < uiCuSize; ++i)
    {
      std::cout << ppsCusLuma[uiCuIdx][i] << ',';
    }
    std::cout << std::endl;
  }

  return pIsSplit;
}