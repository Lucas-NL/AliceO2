// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <sstream>
#include <string>
#include "EMCALReconstruction/RawReaderMemory.h"
#include "EMCALReconstruction/RawDecodingError.h"
#include "DetectorsRaw/RDHUtils.h"

using namespace o2::emcal;

using RDHDecoder = o2::raw::RDHUtils;

RawReaderMemory::RawReaderMemory(gsl::span<const char> rawmemory) : mRawMemoryBuffer(rawmemory)
{
  init();
}

void RawReaderMemory::setRawMemory(const gsl::span<const char> rawmemory)
{
  mRawMemoryBuffer = rawmemory;
  init();
}

o2::header::RDHAny RawReaderMemory::decodeRawHeader(const void* payloadwords)
{
  auto headerversion = RDHDecoder::getVersion(payloadwords);
  if (headerversion == 4) {
    return o2::header::RDHAny(*reinterpret_cast<const o2::header::RAWDataHeaderV4*>(payloadwords));
  } else if (headerversion == 5) {
    return o2::header::RDHAny(*reinterpret_cast<const o2::header::RAWDataHeaderV5*>(payloadwords));
  } else if (headerversion == 6) {
    return o2::header::RDHAny(*reinterpret_cast<const o2::header::RAWDataHeaderV6*>(payloadwords));
  }
  throw RawDecodingError(RawDecodingError::ErrorType_t::HEADER_DECODING);
}

void RawReaderMemory::init()
{
  mCurrentPosition = 0;
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  mRawBuffer.flush();
  mNumData = mRawMemoryBuffer.size() / 8192; // assume fixed 8 kB pages
}

void RawReaderMemory::next()
{
  mRawPayload.reset();
  mCurrentTrailer.reset();
  bool isDataTerminated = false;
  do {
    nextPage(false);
    if (hasNext()) {
      auto nextheader = decodeRawHeader(mRawMemoryBuffer.data() + mCurrentPosition);
      /**
       * eventually in the future check continuing payload based on the bc/orbit ID
      auto currentbc = RDHDecoder::getTriggerBC(mRawHeader),
           nextbc = RDHDecoder::getTriggerBC(nextheader);
      auto currentorbit = RDHDecoder::getTriggerOrbit(mRawHeader),
           nextorbit = RDHDecoder::getTriggerOrbit(nextheader);
      **/
      auto nextpagecounter = RDHDecoder::getPageCounter(nextheader);
      if (nextpagecounter == 0) {
        isDataTerminated = true;
      } else
        isDataTerminated = false;
    } else
      isDataTerminated = true;
    // Check if the data continues
  } while (!isDataTerminated);
  // add combined trailer to payload
  mRawPayload.appendPayloadWords(mCurrentTrailer.encode());
}

void RawReaderMemory::nextPage(bool doResetPayload)
{
  if (!hasNext())
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAGE_NOTFOUND);
  if (doResetPayload)
    mRawPayload.reset();
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  // Read header
  try {
    mRawHeader = decodeRawHeader(mRawMemoryBuffer.data() + mCurrentPosition);
    RDHDecoder::printRDH(mRawHeader);
    mRawHeaderInitialized = true;
  } catch (...) {
    throw RawDecodingError(RawDecodingError::ErrorType_t::HEADER_DECODING);
  }
  if (mCurrentPosition + RDHDecoder::getMemorySize(mRawHeader) >= mRawMemoryBuffer.size()) {
    // Payload incomplete
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAYLOAD_DECODING);
  } else {
    mRawBuffer.readFromMemoryBuffer(gsl::span<const char>(mRawMemoryBuffer.data() + mCurrentPosition + RDHDecoder::getHeaderSize(mRawHeader), RDHDecoder::getMemorySize(mRawHeader) - RDHDecoder::getHeaderSize(mRawHeader)));

    // Read off and chop trailer
    //
    // Every page gets a trailer. The trailers from the single pages need to be removed.
    // There will be a combined trailer which keeps the sum of the payloads for all trailers.
    // This will be appended to the chopped payload.
    auto trailer = RCUTrailer::constructFromPayloadWords(mRawBuffer.getDataWords());
    if (!mCurrentTrailer.isInitialized()) {
      mCurrentTrailer = trailer;
    } else {
      mCurrentTrailer.setPayloadSize(mCurrentTrailer.getPayloadSize() + trailer.getPayloadSize());
    }
    gsl::span<const uint32_t> payloadWithoutTrailer(mRawBuffer.getDataWords().data(), mRawBuffer.getDataWords().size() - trailer.getTrailerSize());

    mRawPayload.appendPayloadWords(payloadWithoutTrailer);
    mRawPayload.increasePageCount();
  }
  mCurrentPosition += RDHDecoder::getOffsetToNext(mRawHeader); /// Assume fixed 8 kB page size
}

void RawReaderMemory::readPage(int page)
{
  int currentposition = 8192 * page;
  if (currentposition >= mRawMemoryBuffer.size())
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAGE_NOTFOUND);
  mRawHeaderInitialized = false;
  mPayloadInitialized = false;
  // Read header
  try {
    mRawHeader = decodeRawHeader(mRawMemoryBuffer.data() + mCurrentPosition);
    mRawHeaderInitialized = true;
  } catch (...) {
    throw RawDecodingError(RawDecodingError::ErrorType_t::HEADER_DECODING);
  }
  if (currentposition + RDHDecoder::getHeaderSize(mRawHeader) + RDHDecoder::getMemorySize(mRawHeader) >= mRawMemoryBuffer.size()) {
    // Payload incomplete
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAYLOAD_DECODING);
  } else {
    mRawBuffer.readFromMemoryBuffer(gsl::span<const char>(mRawMemoryBuffer.data() + currentposition + RDHDecoder::getHeaderSize(mRawHeader), RDHDecoder::getMemorySize(mRawHeader)));
  }
}

const o2::header::RDHAny& RawReaderMemory::getRawHeader() const
{
  if (!mRawHeaderInitialized)
    throw RawDecodingError(RawDecodingError::ErrorType_t::HEADER_INVALID);
  return mRawHeader;
}

const RawBuffer& RawReaderMemory::getRawBuffer() const
{
  if (!mPayloadInitialized)
    throw RawDecodingError(RawDecodingError::ErrorType_t::PAYLOAD_INVALID);
  return mRawBuffer;
}
