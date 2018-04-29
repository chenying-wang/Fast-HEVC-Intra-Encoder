#include <iostream>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

#include "SessionWrapper.h"

using namespace tensorflow;

/**
 * Constructor
*/
SessionWrapper::SessionWrapper()
{
  Status status = NewSession(SessionOptions(), &m_pcSession);
  if (!status.ok())
  {
    std::cout << status.ToString() << std::endl;
  }
}

/**
 * Destructor
*/
SessionWrapper::~SessionWrapper()
{
  m_pcSession -> Close();

  delete[] m_pbIsSplit;
}

/**
 * Initialize
*/
Void SessionWrapper::init(UInt uiNumOfCus)
{
  m_pbIsSplit = new Bool[uiNumOfCus];

  // tensorflow::GraphDef graph_def;
  // tensorflow::Status status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), "../../cnn/.tmp/frozen_modle.pb", &graph_def);
  // if (!status.ok()) {
  //   std::cout << status.ToString() << std::endl;
  //   return;
  // }
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
