#include "src/pjrt/api/pjrt_c_api.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

extern "C" const PJRT_Api* GetPjrtApi();

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool TypeOf(const std::string& code, PJRT_Buffer_Type* t) {
  if (code == "f32")  { *t = PJRT_Buffer_Type_F32;  return true; }
  if (code == "f16")  { *t = PJRT_Buffer_Type_F16;  return true; }
  if (code == "i32")  { *t = PJRT_Buffer_Type_S32;  return true; }
  if (code == "i8")   { *t = PJRT_Buffer_Type_S8;   return true; }
  if (code == "u8")   { *t = PJRT_Buffer_Type_U8;   return true; }
  if (code == "i16")  { *t = PJRT_Buffer_Type_S16;  return true; }
  if (code == "u16")  { *t = PJRT_Buffer_Type_U16;  return true; }
  if (code == "u32")  { *t = PJRT_Buffer_Type_U32;  return true; }
  if (code == "pred") { *t = PJRT_Buffer_Type_PRED; return true; }
  return false;
}

std::vector<int64_t> ParseShape(const std::string& s) {
  std::vector<int64_t> dims;
  std::string cur;
  for (char c : s) {
    if (c == 'x') { if (!cur.empty()) { dims.push_back(std::stoll(cur)); cur.clear(); } }
    else cur.push_back(c);
  }
  if (!cur.empty()) dims.push_back(std::stoll(cur));
  return dims;
}

const PJRT_Api* g_api = nullptr;

int Fail(PJRT_Error* err, const char* what) {
  PJRT_Error_Message_Args m;
  std::memset(&m, 0, sizeof(m));
  m.struct_size = sizeof(m);
  m.error = err;
  g_api->PJRT_Error_Message(&m);
  fprintf(stderr, "pjrt_e2e: %s: %.*s\n", what, (int)m.message_size, m.message);
  return 3;
}

}

int main(int argc, char** argv) {
  if (argc < 3 || (argc - 3) % 3 != 0) {
    fprintf(stderr, "usage: %s <artifact.mlirbc> <out_dir> [<in.bin> <typecode> <shape>]...\n", argv[0]);
    return 2;
  }
  std::string artifact = ReadFile(argv[1]);
  std::string out_dir = argv[2];
  if (artifact.empty()) { fprintf(stderr, "pjrt_e2e: empty artifact\n"); return 2; }

  g_api = GetPjrtApi();

  PJRT_Client_Create_Args cc;
  std::memset(&cc, 0, sizeof(cc));
  cc.struct_size = sizeof(cc);
  if (PJRT_Error* e = g_api->PJRT_Client_Create(&cc)) return Fail(e, "Client_Create");
  PJRT_Client* client = cc.client;

  PJRT_Client_AddressableDevices_Args da;
  std::memset(&da, 0, sizeof(da));
  da.struct_size = sizeof(da);
  da.client = client;
  if (PJRT_Error* e = g_api->PJRT_Client_AddressableDevices(&da)) return Fail(e, "AddressableDevices");
  if (da.num_addressable_devices == 0) { fprintf(stderr, "pjrt_e2e: no devices\n"); return 3; }
  PJRT_Device* device = da.addressable_devices[0];

  int n_in = (argc - 3) / 3;
  std::vector<std::string> in_bytes(n_in);
  std::vector<std::vector<int64_t>> in_dims(n_in);
  std::vector<PJRT_Buffer*> in_bufs(n_in);
  for (int i = 0; i < n_in; ++i) {
    std::string path = argv[3 + i * 3];
    std::string code = argv[4 + i * 3];
    in_dims[i] = ParseShape(argv[5 + i * 3]);
    PJRT_Buffer_Type type;
    if (!TypeOf(code, &type)) { fprintf(stderr, "pjrt_e2e: bad typecode '%s'\n", code.c_str()); return 2; }
    in_bytes[i] = ReadFile(path);

    PJRT_Client_BufferFromHostBuffer_Args fb;
    std::memset(&fb, 0, sizeof(fb));
    fb.struct_size = sizeof(fb);
    fb.client = client;
    fb.data = in_bytes[i].data();
    fb.type = type;
    fb.dims = in_dims[i].data();
    fb.num_dims = in_dims[i].size();
    fb.device = device;
    if (PJRT_Error* e = g_api->PJRT_Client_BufferFromHostBuffer(&fb)) return Fail(e, "BufferFromHostBuffer");
    in_bufs[i] = fb.buffer;
    if (fb.done_with_host_buffer) {
      PJRT_Event_Destroy_Args ed; std::memset(&ed, 0, sizeof(ed)); ed.struct_size = sizeof(ed);
      ed.event = fb.done_with_host_buffer; g_api->PJRT_Event_Destroy(&ed);
    }
  }

  PJRT_Program prog;
  std::memset(&prog, 0, sizeof(prog));
  prog.struct_size = sizeof(prog);
  prog.code = const_cast<char*>(artifact.data());
  prog.code_size = artifact.size();
  static const char kFmt[] = "mlir";
  prog.format = kFmt;
  prog.format_size = std::strlen(kFmt);

  PJRT_Client_Compile_Args ca;
  std::memset(&ca, 0, sizeof(ca));
  ca.struct_size = sizeof(ca);
  ca.client = client;
  ca.program = &prog;
  if (PJRT_Error* e = g_api->PJRT_Client_Compile(&ca)) return Fail(e, "Client_Compile");
  PJRT_LoadedExecutable* loaded = ca.executable;

  PJRT_LoadedExecutable_GetExecutable_Args ge;
  std::memset(&ge, 0, sizeof(ge));
  ge.struct_size = sizeof(ge);
  ge.loaded_executable = loaded;
  if (PJRT_Error* e = g_api->PJRT_LoadedExecutable_GetExecutable(&ge)) return Fail(e, "GetExecutable");
  PJRT_Executable_NumOutputs_Args no;
  std::memset(&no, 0, sizeof(no));
  no.struct_size = sizeof(no);
  no.executable = ge.executable;
  if (PJRT_Error* e = g_api->PJRT_Executable_NumOutputs(&no)) return Fail(e, "NumOutputs");
  int n_out = (int)no.num_outputs;

  std::vector<PJRT_Buffer*> out_bufs(n_out, nullptr);
  PJRT_Buffer* const* arg_row = in_bufs.data();
  PJRT_Buffer** out_row = out_bufs.data();

  PJRT_ExecuteOptions opts;
  std::memset(&opts, 0, sizeof(opts));
  opts.struct_size = sizeof(opts);

  PJRT_LoadedExecutable_Execute_Args ex;
  std::memset(&ex, 0, sizeof(ex));
  ex.struct_size = sizeof(ex);
  ex.executable = loaded;
  ex.options = &opts;
  ex.num_devices = 1;
  ex.num_args = n_in;
  ex.argument_lists = &arg_row;
  ex.output_lists = &out_row;
  ex.execute_device = device;
  if (PJRT_Error* e = g_api->PJRT_LoadedExecutable_Execute(&ex)) return Fail(e, "Execute");

  for (int i = 0; i < n_out; ++i) {
    PJRT_Buffer_ToHostBuffer_Args th;
    std::memset(&th, 0, sizeof(th));
    th.struct_size = sizeof(th);
    th.src = out_bufs[i];
    if (PJRT_Error* e = g_api->PJRT_Buffer_ToHostBuffer(&th)) return Fail(e, "ToHostBuffer(size)");
    std::vector<uint8_t> host(th.dst_size);
    th.dst = host.data();
    if (PJRT_Error* e = g_api->PJRT_Buffer_ToHostBuffer(&th)) return Fail(e, "ToHostBuffer(copy)");

    std::ostringstream p;
    p << out_dir << "/out" << i << ".bin";
    std::ofstream f(p.str(), std::ios::binary);
    f.write((const char*)host.data(), (std::streamsize)host.size());
  }

  fprintf(stderr, "pjrt_e2e: %d inputs -> %d outputs via PJRT Compile+Execute\n", n_in, n_out);
  return 0;
}
