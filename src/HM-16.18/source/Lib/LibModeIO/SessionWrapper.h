#include <iostream>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

#include "TLibCommon/CommonDef.h"

using namespace tensorflow;

#ifndef __SESSIONWRAPPER__
#define __SESSIONWRAPPER__

class SessionWrapper
{
private:
  Session **m_ppcSession;
  GraphDef **m_ppcGraphDef;
  Bool *m_pbIsSplit;
  UInt *m_puiCuSize;

  UChar m_uhTotalDepth;

public:
  SessionWrapper();
  virtual ~SessionWrapper();
  Void init(UInt uiNumOfCus, UInt uiLogMaxCuWidth, UChar uhTotalDepth);
  Bool *infer(Pel **ppsCusLuma, UInt uiNumOfCus, UChar uhDepth);
};

#endif // __SESSIONWRAPPER__
