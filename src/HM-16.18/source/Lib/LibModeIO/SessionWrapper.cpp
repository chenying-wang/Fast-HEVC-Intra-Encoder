#include <iostream>

#include "SessionWrapper.h"
// #include "tensorflow/core/platform/env.h"
// #include "tensorflow/core/public/session.h"

// using namespace tensorflow;
// using namespace tensorflow::ops;

/**
 * Constructor
*/
SessionWrapper::SessionWrapper()
{

}

/**
 * Destructor
*/
SessionWrapper::~SessionWrapper()
{
  delete[] m_pbIsSplit;
}

Void SessionWrapper::init(UInt uiNumOfCus)
{
  m_pbIsSplit = new Bool[uiNumOfCus];
}

/**
 * Call TensorFlow to infer if split the remaining CTUs/CUs in a particular depth
*/
Bool *SessionWrapper::infer(Pel **ppsCusLuma, UInt uiNumOfCus, UChar uhDepth)
{
  UInt uiCuSize = 1 << ((6 - uhDepth) << 1);

  // To-Do: sess.run();

  for (UInt uiCuIdx = 0; uiCuIdx < uiNumOfCus; ++uiCuIdx)
  {
    m_pbIsSplit[uiCuSize] = false;
  }

  return m_pbIsSplit;
}
