//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef PTI_UTILS_ELF_PARSER_H_
#define PTI_UTILS_ELF_PARSER_H_

#include <string.h>

#include <vector>

#include "elf.h"
#include "debug_line_parser.h"
#include "debug_info_parser.h"
#include "debug_abbrev_parser.h"

class ElfParser {
 public:
  ElfParser(const uint8_t* data, uint32_t size) : data_(data), size_(size) {}

  bool IsValid() const {
    if (data_ == nullptr || size_ < sizeof(Elf64Header)) {
      return false;
    }

    const Elf64Header* header = reinterpret_cast<const Elf64Header*>(data_);
    if (header->ident[0] != ELF_MAGIC_NUMBER ||
        header->ident[1] != 'E' ||
        header->ident[2] != 'L' ||
        header->ident[3] != 'F') {
      return false;
    }


    if (header->ident[4] != 2) { // 64-bit format
      return false;
    }

    return true;
  }

  std::vector<std::string> GetFileList() const {
    if (!IsValid()) {
      return std::vector<std::string>();
    }

    const uint8_t* section = nullptr;
    uint64_t section_size = 0;
    GetSection(".debug_line", &section, &section_size);
    if (section == nullptr || section_size == 0) {
      return std::vector<std::string>();
    }

    PTI_ASSERT(section_size < (std::numeric_limits<uint32_t>::max)());
    DebugLineParser line_parser(section, static_cast<uint32_t>(section_size));
    if (!line_parser.IsValid()) {
      return std::vector<std::string>();
    }

    GetSection(".debug_abbrev", &section, &section_size);
    if (section == nullptr || section_size == 0) {
      return std::vector<std::string>();
    }

    PTI_ASSERT(section_size < (std::numeric_limits<uint32_t>::max)());
    DebugAbbrevParser abbrev_parser(section, static_cast<uint32_t>(section_size));
    if (!abbrev_parser.IsValid()) {
      return std::vector<std::string>();
    }

    DwarfCompUnitMap comp_unit_map = abbrev_parser.GetCompUnitMap();
    if (comp_unit_map.size() == 0) {
      return std::vector<std::string>();
    }

    GetSection(".debug_info", &section, &section_size);
    if (section == nullptr || section_size == 0) {
      return std::vector<std::string>();
    }

    PTI_ASSERT(section_size < (std::numeric_limits<uint32_t>::max)());
    DebugInfoParser info_parser(section, static_cast<uint32_t>(section_size));
    if (!info_parser.IsValid()) {
      return std::vector<std::string>();
    }

    std::vector<std::string> file_path_list;
    std::vector<FileInfo> file_list = line_parser.GetFileList();
    std::vector<std::string> dir_list = line_parser.GetDirList();
    for (size_t i = 0; i < file_list.size(); ++i) {
      uint32_t path_index = file_list[i].path_index;
      PTI_ASSERT(path_index <= dir_list.size());
      if (path_index == 0) {
        std::string comp_dir = info_parser.GetCompDir(comp_unit_map);
        if (!comp_dir.empty()) {
          file_path_list.push_back(comp_dir + "/" + file_list[i].name);
        } else {
          file_path_list.push_back(file_list[i].name);
        }
      } else {
        file_path_list.push_back(dir_list[path_index - 1] + "/" + file_list[i].name);
      }
    }

    return file_path_list;
  }

  std::vector<LineInfo> GetLineInfo() const {
    if (!IsValid()) {
      return std::vector<LineInfo>();    
    }

    const uint8_t* section = nullptr;
    uint64_t section_size = 0;
    GetSection(".debug_line", &section, &section_size);
    if (section == nullptr || section_size == 0) {
      return std::vector<LineInfo>();
    }

    PTI_ASSERT(section_size < (std::numeric_limits<uint32_t>::max)());
    DebugLineParser parser(section, static_cast<uint32_t>(section_size));
    if (!parser.IsValid()) {
      return std::vector<LineInfo>();
    }

    return parser.GetLineInfo();
  }

  std::vector<uint8_t> GetGenBinary() const {
    if (!IsValid()) {
      return std::vector<uint8_t>();
    }

    const uint8_t* section = nullptr;
    uint64_t section_size = 0;
    GetSection("Intel(R) OpenCL Device Binary", &section, &section_size);
    if (section == nullptr || section_size == 0) {
      return std::vector<uint8_t>();
    }

    std::vector<uint8_t> binary(section_size);
    memcpy(binary.data(), section, section_size);
    return binary;
  }

 private:
  void GetSection(const char* name,
                  const uint8_t** section,
                  uint64_t* section_size) const {
    PTI_ASSERT(section != nullptr && section_size != nullptr);

    const Elf64Header* header = reinterpret_cast<const Elf64Header*>(data_);
    const Elf64SectionHeader* section_header =
      reinterpret_cast<const Elf64SectionHeader*>(data_ + header->shoff);
    const char* name_section = reinterpret_cast<const char*>(
        data_ + section_header[header->shstrndx].offset);
    
    for (uint32_t i = 1; i < header->shnum; ++i) {
      const char* section_name = name_section + section_header[i].name;
      if (strcmp(section_name, name) == 0) {
        *section = data_ + section_header[i].offset;
        *section_size = section_header[i].size;
        return;
      }
    }

    *section = nullptr;
    *section_size = 0;
  }

 private:
  const uint8_t* data_ = nullptr;
  uint32_t size_ = 0;
};

#endif // PTI_UTILS_ELF_PARSER_H_