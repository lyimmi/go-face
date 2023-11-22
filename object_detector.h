#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
	IMAGE_LOAD_ERROR,
	SERIALIZATION_ERROR,
	UNKNOWN_ERROR,
} err_code;

typedef struct objrec {
	void* cls;
	const char* err_str;
	err_code err_code;
} objrec;

typedef struct objret {
	long* rectangles;
	int rectCount;
	const char* err_str;
	err_code err_code;
} objret;

objrec* objrec_init(const char* model_dir, int len);
objret* objrec_recognize(objrec* rec, const uint8_t* img_data, int len);
void objrec_free(objrec* rec);

#ifdef __cplusplus
}
#endif
