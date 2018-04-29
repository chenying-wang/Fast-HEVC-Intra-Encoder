#include <iostream>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

#include "TLibCommon/CommonDef.h"

#ifndef __SESSIONWRAPPER__
#define __SESSIONWRAPPER__

class SessionWrapper
{
private:
  tensorflow::Session *m_pcSession;
  Bool *m_pbIsSplit;

public:
  SessionWrapper();
  virtual ~SessionWrapper();
  Void init(UInt uiNumOfCus);
  Bool *infer(Pel **ppsCusLuma, UInt uiNumOfCus, UChar uhDepth);
};

#endif // __SESSIONWRAPPER__
