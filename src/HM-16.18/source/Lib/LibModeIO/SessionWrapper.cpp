#include <iostream>
#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
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
  
}

/**
 * Destructor
*/
SessionWrapper::~SessionWrapper()
{
  delete[] m_pbIsSplit;
  delete[] m_puiCuSize;
  if (m_ppcSession != NULL)
  {
    for (UChar uhDepth = 0; uhDepth < m_uhTotalDepth; ++uhDepth)
    {
      m_ppcSession[uhDepth]->Close();
      delete m_ppcGraphDef[uhDepth];
    }
    delete[] m_ppcSession;
    delete[] m_ppcGraphDef;
  }
}

/**
 * Initialize
*/
Void SessionWrapper::init(const UInt uiNumOfCus, const UInt uiLogMaxCuWidth, const UChar uhTotalDepth)
{
  m_uhTotalDepth = uhTotalDepth;
  
  m_pbIsSplit = new Bool[uiNumOfCus];
  m_puiCuSize = new UInt[m_uhTotalDepth];
  for (UChar uhDepth = 0; uhDepth < m_uhTotalDepth; ++uhDepth)
  {
    m_puiCuSize[uhDepth] = 1 << ((uiLogMaxCuWidth - uhDepth) << 1);
  }

  m_ppcSession = new Session*[m_uhTotalDepth];
  m_ppcGraphDef = new GraphDef*[m_uhTotalDepth];
  for (UChar uhDepth = 0; uhDepth < m_uhTotalDepth; ++uhDepth)
  {
    Status status = NewSession(SessionOptions(), m_ppcSession + uhDepth);
    if (!status.ok())
    {
      std::cerr << status.ToString() << std::endl;
      return;
    }

    auto filename = string("../cnn/.tmp/") + char(uhDepth + '0') + string("/frozen_graph.pb");
    m_ppcGraphDef[uhDepth] = new GraphDef();
    status = ReadBinaryProto(Env::Default(), filename, m_ppcGraphDef[uhDepth]);
    if (!status.ok())
    {
      std::cerr << status.ToString() << std::endl;
      return;
    }

    status = m_ppcSession[uhDepth]->Create(*m_ppcGraphDef[uhDepth]);
    if (!status.ok())
    {
      std::cerr << status.ToString() << std::endl;
      return;
    }
  }
}

/**
 * Call TensorFlow to infer if split the remaining CTUs/CUs in a particular depth
*/
Bool *SessionWrapper::infer(Pel **ppsCusLuma, UInt uiNumOfCus, UChar uhDepth)
{
  UInt uiCuSize = m_puiCuSize[uhDepth];
  auto features_tensor = Tensor( DT_FLOAT, TensorShape({uiNumOfCus, uiCuSize}) );
  auto features_tensor_map = features_tensor.tensor<float, 2>();
  for (UInt uiCuIdx = 0; uiCuIdx < uiNumOfCus; ++uiCuIdx)
  {
    for (UInt i = 0; i < uiCuSize; ++i)
    {
      features_tensor_map(uiCuIdx, i) = ppsCusLuma[uiCuIdx][i];
    }
  }

  std::vector<std::pair<string, Tensor>> inputs = {{"input/raw_features", features_tensor}};
  std::vector<Tensor> outputs;

  Status status = m_ppcSession[uhDepth]->Run(inputs, {"softmax/softmax"}, {}, &outputs);
  if (!status.ok())
  {
      std::cerr << "ERROR: prediction failed..." << status.ToString() << std::endl;
      return NULL;
  }

  auto outputs_tensor_map = outputs[0].tensor<float, 2>();
  for (UInt uiCuIdx = 0; uiCuIdx < uiNumOfCus; ++uiCuIdx)
  {
    m_pbIsSplit[uiCuIdx] = (outputs_tensor_map(uiCuIdx, 0) - outputs_tensor_map(uiCuIdx, 1) < 0);
  }
  return m_pbIsSplit;
}
