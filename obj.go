package face

/*
#cgo CXXFLAGS: -std=c++1z -Wall -O3 -DNDEBUG -march=native
#cgo LDFLAGS: -ldlib -lblas -lcblas -llapack -ljpeg
#include <stdlib.h>
#include <stdint.h>
#include "object_detector.h"
*/
import "C"

import (
	"image"
	"io"
	"os"
	"unsafe"
)

type ObjRecognizer struct {
	ptr *C.objrec
}

// Close frees resources taken by the Recognizer. Safe to call multiple
// times. Don't use Recognizer after close call.
func (rec *ObjRecognizer) Close() {
	C.objrec_free(rec.ptr)
	rec.ptr = nil
}

// NewObjRecognizer returns a new recognizer interface. modelDir points to
// directory with shape_detector.svm
func NewObjRecognizer(modelDir string, len int) (rec *ObjRecognizer, err error) {
	cModelDir := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cModelDir))
	ptr := C.objrec_init(cModelDir, C.int(len))

	if ptr.err_str != nil {
		defer C.objrec_free(ptr)
		defer C.free(unsafe.Pointer(ptr.err_str))
		err = makeError(C.GoString(ptr.err_str), int(ptr.err_code))
		return
	}

	rec = &ObjRecognizer{ptr}
	return
}

func (rec *ObjRecognizer) recognizeFile(imgPath string) (rect []image.Rectangle, err error) {
	fd, err := os.Open(imgPath)
	if err != nil {
		return
	}
	defer fd.Close()
	imgData, err := io.ReadAll(fd)
	if err != nil {
		return
	}
	return rec.recognize(imgData)
}

func (rec *ObjRecognizer) Recognize(imgData []byte) (rect []image.Rectangle, err error) {
	return rec.recognize(imgData)
}

func (rec *ObjRecognizer) recognize(imgData []byte) (rect []image.Rectangle, err error) {
	if len(imgData) == 0 {
		err = ImageLoadError("Empty image")
		return
	}
	cImgData := (*C.uint8_t)(&imgData[0])
	cLen := C.int(len(imgData))
	ret := C.objrec_recognize(rec.ptr, cImgData, cLen)
	defer C.free(unsafe.Pointer(ret))

	retCount := int(ret.rectCount)
	if retCount <= 0 {
		return
	}

	rDataLen := retCount * rectLen
	rDataPtr := unsafe.Pointer(ret.rectangles)
	rData := (*[maxElements]C.long)(rDataPtr)[:rDataLen:rDataLen]

	for i := 0; i < retCount; i++ {
		x0 := int(rData[i*rectLen])
		y0 := int(rData[i*rectLen+1])
		x1 := int(rData[i*rectLen+2])
		y1 := int(rData[i*rectLen+3])
		rect = append(rect, image.Rect(x0, y0, x1, y1))
	}
	return
}
