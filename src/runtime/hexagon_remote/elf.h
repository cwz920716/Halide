#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

// ELF comes in 32 and 64-bit variants. Define ELF64 to use the 64-bit
// variant.
// #define ELF64

#ifdef ELF64
typedef uint64_t elfaddr_t;
#else
typedef uint32_t elfaddr_t;
#endif

// The standard ELF header. See
// http://man7.org/linux/man-pages/man5/elf.5.html for the meanings of
// these fields.
struct elf_header_t {
    unsigned char e_ident[16];
    uint16_t      e_type;
    uint16_t      e_machine;
    uint32_t      e_version;
    elfaddr_t     e_entry;
    elfaddr_t     e_phoff;
    elfaddr_t     e_shoff;
    uint32_t      e_flags;
    uint16_t      e_ehsize;
    uint16_t      e_phentsize;
    uint16_t      e_phnum;
    uint16_t      e_shentsize;
    uint16_t      e_shnum;
    uint16_t      e_shstrndx;
};

// An elf section header
struct section_header_t {
    uint32_t   sh_name;
    uint32_t   sh_type;
    elfaddr_t  sh_flags;
    elfaddr_t  sh_addr;
    elfaddr_t  sh_offset;
    elfaddr_t  sh_size;
    uint32_t   sh_link;
    uint32_t   sh_info;
    elfaddr_t   sh_addralign;
    elfaddr_t   sh_entsize;
};

// A symbol table entry
struct symbol_t {
    uint32_t      st_name;

#ifdef ELF64
    unsigned char st_info;
    unsigned char st_other;
    uint16_t      st_shndx;
    elfaddr_t     st_value;
    uint64_t      st_size;
#else
    elfaddr_t     st_value;
    uint32_t      st_size;
    unsigned char st_info;
    unsigned char st_other;
    uint16_t      st_shndx;
#endif
};

// A relocation from a relocation section
struct rela_t {
    elfaddr_t  r_offset;
#ifdef ELF64
    uint64_t   r_info;
    uint32_t   r_type() {return r_info & 0xffffffff;}
    uint32_t   r_sym()  {return r_info >> 32;}
    int64_t    r_addend;
#else
    uint32_t   r_info;
    uint32_t   r_type() {return r_info & 0xff;}
    uint32_t   r_sym()  {return r_info >> 8;}
    int32_t    r_addend;
#endif
};

// Some external symbols we to resolve during relocations
extern "C" int __hexagon_muldf3;
extern "C" int __hexagon_divdf3;
extern "C" int __hexagon_adddf3;
extern "C" int __hexagon_divsf3;
extern "C" int __hexagon_udivdi3;
extern "C" int qurt_hvx_lock(int);
extern "C" int qurt_hvx_unlock();

struct elf_t {
    // The object file in memory
    char *buf;
    size_t size;

    // Set to true to spew debug info
    bool debug;

    // Pointer to the header
    elf_header_t *header;

    // Sections of interest
    section_header_t
        *sec_symtab,   // The symbol table
        *sec_secnames, // The name of each section, i.e. the table of contents
        *sec_text,     // The .text section where the functions live
        *sec_strtab;   // The string table, for looking up symbol names

    // Point to an object file loaded into memory. Does not take
    // ownership of the memory. Memory should be page-aligned.
    void parse_object_file(char *b, size_t s, bool d = false) {
        buf = b;
        size = s;
        debug = d;

        // Grab the header
        if (size < sizeof(elf_header_t)) abort();
        header = (elf_header_t *)buf;

        // Get the section names section first
        sec_secnames = get_section(header->e_shstrndx);

        // Walk over the other sections
        for (int i = 0; i < header->e_shnum; i++) {
            section_header_t *sec = get_section(i);
            const char *sec_name = get_section_name(sec);
            if (debug) printf("\nSection %s at %p:\n",
                              sec_name,
                              get_addr(get_section_offset(sec)));

            if (sec->sh_type == 2) {
                sec_symtab = sec;
            } else if (sec->sh_type == 3) {
                sec_strtab = sec;
            } else if (sec->sh_type == 1 && strncmp(sec_name, ".text", 5) == 0) {
                sec_text = sec;
            }
        }
    }

    // Get the address given an offset into the buffer. Asserts that
    // it's in-range.
    char *get_addr(elfaddr_t off) {
        // elfaddr_t is unsigned
        if (off >= size) abort();
        return buf + off;
    }

    // Get the number of sections
    int num_sections() {
        return header->e_shnum;
    }

    // Get a section by index
    section_header_t *get_section(int i) {
        if (!header) abort();
        elfaddr_t off = header->e_shoff + i * header->e_shentsize;
        if (off + sizeof(section_header_t) > size) abort();
        return (section_header_t *)get_addr(off);
    }

    // Get the starting address of a section
    char *get_section_start(section_header_t *sec) {
        return get_addr(sec->sh_offset);
    }

    // Get the offset of a section
    elfaddr_t get_section_offset(section_header_t *sec) {
        return sec->sh_offset;
    }

    // Get the size of a section in bytes
    int get_section_size(section_header_t *sec) {
        return sec->sh_size;
    }

    // Get the name of a section
    const char *get_section_name(section_header_t *sec) {
        if (!sec_secnames) abort();
        return get_addr(get_section_offset(sec_secnames) + sec->sh_name);
    }

    // Look up a section by name
    section_header_t *find_section(const char *name) {
        for (int i = 0; i < num_sections(); i++) {
            section_header_t *sec = get_section(i);
            const char *sec_name = get_section_name(sec);
            if (strncmp(sec_name, name, strlen(name)+1) == 0) {
                return sec;
            }
        }
        return NULL;
    }

    // The number of symbols in the symbol table
    int num_symbols() {
        if (!sec_symtab) abort();
        return get_section_size(sec_symtab) / sizeof(symbol_t);
    }

    // Get a symbol from the symbol table by index
    symbol_t *get_symbol(int i) {
        if (!sec_symtab) abort();
        return (symbol_t *)(get_addr(get_section_offset(sec_symtab) + i * sizeof(symbol_t)));
    }

    // Get the name of a symbol
    const char *get_symbol_name(symbol_t *sym) {
        if (!sec_strtab) abort();
        return (const char *)(get_addr(get_section_offset(sec_strtab) + sym->st_name));
    }

    // Get the section a symbol exists in. NULL for extern symbols.
    section_header_t *get_symbol_section(symbol_t *sym) {
        if (sym->st_shndx == 0) return NULL;
        return get_section(sym->st_shndx);
    }

    // Check if a symbol exists in this object file
    bool symbol_is_defined(symbol_t *sym) {
        return get_symbol_section(sym) != NULL;
    }

    // Get the address of a symbol
    char *get_symbol_addr(symbol_t *sym) {
        return get_addr(get_section_offset(get_symbol_section(sym)) + sym->st_value);
    }

    // Look up a symbol by name
    symbol_t *find_symbol(const char *name) {
        const size_t len = strlen(name);

        for (int i = 0; i < num_symbols(); i++) {
            symbol_t *sym = get_symbol(i);
            const char *sym_name = get_symbol_name(sym);
            if (strncmp(sym_name, name, len+1) == 0) {
                return sym;
            }
        }

        return NULL;
    }

    // Get the number of relocations in a relocation section
    int num_relas(section_header_t *sec_rela) {
        if (!sec_rela) abort();
        return get_section_size(sec_rela) / sizeof(rela_t);
    }

    // Get a relocation from a relocation section by index
    rela_t *get_rela(section_header_t *sec_rela, int i) {
        if (!sec_rela) abort();
        return (rela_t *)(get_addr(get_section_offset(sec_rela) + i * sizeof(rela_t)));
    }

    // Perform a single relocation.
    void do_reloc(char *addr, uint32_t mask, uintptr_t val) {
        uint32_t inst = *((uint32_t *)addr);
        if (debug) {
            printf("Fixup inside instruction at %lx:\n  %08lx\n",
                   (uint32_t)(addr - get_addr(get_section_offset(sec_text))), inst);
            printf("val: 0x%08lx\n", (unsigned long)val);
            printf("mask: 0x%08lx\n", mask);
        }

        if (!mask) {

            // The mask depends on the instruction.
            if (debug) {
                // First print the bits so I can search for it in the
                // instruction encodings.
                printf("Instruction bits: ");
                for (int i = 31; i >=0; i--) {
                    printf("%d", (int)((inst >> i) & 1));
                }
                printf("\n");
            }

            if ((inst & (3 << 14)) == 0) {
                // Some instructions are actually pairs of 16-bit
                // subinstructions
                if (debug) printf("Duplex!\n");

                int iclass = ((inst >> 29) << 1) | ((inst >> 13) & 1);
                if (debug) {
                    printf("Class: %x\n", iclass);
                    printf("Hi: ");
                    for (int i = 28; i >= 16; i--) {
                        printf("%d", (int)((inst >> i) & 1));
                    }
                    printf("\n");
                    printf("Lo: ");
                    for (int i = 12; i >= 0; i--) {
                        printf("%d", (int)((inst >> i) & 1));
                    }
                    printf("\n");
                }

                // We only know how to do the ones where the high
                // subinstruction is an immediate assignment. (marked
                // as A in table 9-4 in the programmer's reference
                // manual).
                if (iclass < 3 || iclass > 7) {
                    abort();
                }

                // Pull out the subinstructions. They're the low 13
                // bits of each half-word.
                uint32_t hi = (inst >> 16) & ((1 << 13) - 1);
                uint32_t lo = inst & ((1 << 13) - 1);

                // We only understand the ones where hi starts with 010
                if ((hi >> 10) != 2) {
                    abort();
                }

                // Low 6 bits of val go in the following bits.
                mask = 63 << 20;

            } else if (((inst >> 24) & 249) == 72) {
                // Example instruction encoding that has this high byte (ignoring bits 1 and 2):
                // 0100 1ii0  000i iiii  PPit tttt  iiii iiii
                if (debug) printf("Instruction-specific case A\n");
                mask = 0x061f20ff;
            } else if (((inst >> 24) & 249) == 73) {
                // 0100 1ii1  000i iiii  PPii iiii  iiid dddd
                if (debug) printf("Instruction-specific case B\n");
                mask = 0x061f3fe0;
            } else if ((inst >> 24) == 120) {
                // 0111 1000  ii-i iiii  PPii iiii  iiid dddd
                if (debug) printf("Instruction-specific case C\n");
                mask = 0x00df3fe0;
            } else {
                printf("Unhandled!\n");
                abort();
            }
        }

        for (int i = 0; i < 32; i++) {
            if (mask & (1 << i)) {
                if (inst & (1 << i)) {
                    // This bit should be zero in the unrelocated instruction
                    abort();
                }
                // Consume a bit of val
                int next_bit = val & 1;
                val >>= 1;
                inst |= (next_bit << i);
            }
        }

        if (debug) printf("Relocated instruction:\n  %08lx\n", inst);
        *((uint32_t *)addr) = inst;
    }

    // Do all the relocations for sec (e.g. .text), using the list of
    // relocations in sec_rela (e.g. .rela.text)
    void do_relocations_for_section(section_header_t *sec, section_header_t *sec_rela) {
        if (!sec_rela || !sec) abort();

        struct known_sym {
            const char *name;
            char *addr;
        };
        static known_sym known_syms[] = {
            {"close", (char *)(&close)},
            {"abort", (char *)(&abort)},
            {"memcpy", (char *)(&memcpy)},
            {"memmove", (char *)(&memmove)},
            {"qurt_hvx_lock", (char *)(&qurt_hvx_lock)},
            {"qurt_hvx_unlock", (char *)(&qurt_hvx_unlock)},
            {"__hexagon_divdf3", (char *)(&__hexagon_divdf3)},
            {"__hexagon_muldf3", (char *)(&__hexagon_muldf3)},
            {"__hexagon_adddf3", (char *)(&__hexagon_adddf3)},
            {"__hexagon_divsf3", (char *)(&__hexagon_divsf3)},
            {"__hexagon_udivdi3", (char *)(&__hexagon_udivdi3)},
            {NULL, NULL}
        };

        for (int i = 0; i < num_relas(sec_rela); i++) {
            rela_t *rela = get_rela(sec_rela, i);
            if (debug) printf("\nRelocation %d:\n", i);

            // The location to make a change
            char *fixup_addr = get_addr(get_section_offset(sec) + rela->r_offset);
            if (debug) printf("Fixed address %p\n", fixup_addr);

            // We're fixing up a reference to the following symbol
            symbol_t *sym = get_symbol(rela->r_sym());

            const char *sym_name = get_symbol_name(sym);
            if (debug) printf("Applies to symbol %s\n", sym_name);

            char *sym_addr = NULL;
            if (!symbol_is_defined(sym)) {
                for (int i = 0; known_syms[i].name; i++) {
                    if (strncmp(sym_name, known_syms[i].name, strlen(known_syms[i].name)+1) == 0) {
                        sym_addr = known_syms[i].addr;
                    }
                }
                if (!sym_addr) {
                    // Try dlsym
                    sym_addr = (char *)dlsym(NULL, sym_name);
                }
                if (!sym_addr) {
                    printf("Failed to resolve external symbol: %s\n", sym_name);
                    abort();
                }
            } else {
                section_header_t *sym_sec = get_symbol_section(sym);
                const char *sym_sec_name = get_section_name(sym_sec);
                if (debug) printf("Symbol is in section: %s\n", sym_sec_name);

                sym_addr = get_symbol_addr(sym);
                if (debug) printf("Symbol is at address: %p\n", sym_addr);
            }

            // Define the variables from Table 11-5
            char *S = sym_addr;
            char *P = fixup_addr;
            // TODO: read from the GP register
            char *GP = (char *)0x56000;// get_section_start(find_section(".sdata.4"));
            intptr_t A = rela->r_addend;

            const uint32_t Word32 = 0xffffffff;
            const uint32_t Word32_B22 = 0x01ff3ffe;
            const uint32_t Word32_GP = 0;
            const uint32_t Word32_X26 = 0x0fff3fff;
            const uint32_t Word32_U6 = 0;
            const uint32_t Word32_R6 = 0x000007e0;

            switch (rela->r_type()) {
            case 1:
                do_reloc(fixup_addr, Word32_B22, intptr_t(S + A - P) >> 2);
                break;
            case 6:
                do_reloc(fixup_addr, Word32, intptr_t(S + A) >> 2);
                break;
            case 9:
                do_reloc(fixup_addr, Word32_GP, uintptr_t(S + A - GP));
                break;
            case 10:
                printf("Warning: using bogus GP (%p)\n", GP);
                do_reloc(fixup_addr, Word32_GP, uintptr_t(S + A - GP) >> 1);
                break;
            case 11:
                printf("Warning: using bogus GP (%p)\n", GP);
                do_reloc(fixup_addr, Word32_GP, uintptr_t(S + A - GP) >> 2);
                break;
            case 12:
                printf("Warning: using bogus GP (%p)\n", GP);
                do_reloc(fixup_addr, Word32_GP, uintptr_t(S + A - GP) >> 3);
                break;
            case 17:
                do_reloc(fixup_addr, Word32_X26, uintptr_t(S + A) >> 6);
                break;
            case 23:
                do_reloc(fixup_addr, Word32_U6, uintptr_t(S + A));
                break;
            case 24:
                do_reloc(fixup_addr, Word32_R6, uintptr_t(S + A));
                break;
            case 30:
                do_reloc(fixup_addr, Word32_U6, uintptr_t(S + A));
                break;
            default:
                printf("Unhandled relocation type %lu.\n", rela->r_type());
                abort();
            }
        }

    }

    // Do relocations for all relocation sections in the object file
    void do_relocations() {
        for (int i = 0; i < num_sections(); i++) {
            section_header_t *sec = get_section(i);
            const char *sec_name = get_section_name(sec);
            if (strncmp(sec_name, ".rela.", 6) == 0) {
                // It's a relocations section for something
                section_header_t *sec_to_relocate = find_section(sec_name + 5);
                if (!sec_to_relocate) abort();
                if (debug) printf("Relocating: %s\n", sec_name);
                do_relocations_for_section(sec_to_relocate, sec);
            }
        }
    }

    // Mark the pages of the object file executable
    void make_executable() {
        int err = mprotect(buf, size, PROT_EXEC | PROT_READ | PROT_WRITE);
        if (err) {
            abort();
        }
    }

    // Dump the object file to stdout base-64 encoded
    void dump_as_base64() {
        // For base-64 encoding
        static const char encoding_table[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                                              'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
                                              'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                                              'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                              'w', 'x', 'y', 'z', '0', '1', '2', '3',
                                              '4', '5', '6', '7', '8', '9', '+', '/'};
        // Dump the object in base 64
        printf("BEGIN BASE64\n");
        for (int i = 0; i < size; i += 3) {
            // every group of 3 bytes becomes 4 output bytes
            uint32_t a = buf[i];
            uint32_t b = buf[i+1];
            uint32_t c = buf[i+2];
            uint32_t triple = (a << 16) | (b << 8) | c;
            printf("%c%c%c%c",
                   encoding_table[(triple >> (3*6)) & 0x3f],
                   encoding_table[(triple >> (2*6)) & 0x3f],
                   encoding_table[(triple >> (1*6)) & 0x3f],
                   encoding_table[(triple >> (0*6)) & 0x3f]);
        }
        printf("\nEND BASE64\n");
    }

    /*
    void dump_to_file(const char *f) {
        printf("Dumping to file!\n");
        int fd = open(f, O_CREAT | O_WRONLY);
        write(fd, buf, size);
        close(fd);
    }
    */
};


// Poor man's implementations of dlopen, dlsym, and dlclose using the
// above elf parser. Note that it expects relocatable object files,
// rather than shared objects. Compile them with -fno-pic.
inline void *fake_dlopen(const char *filename, int) {
    int fd = open(filename, O_RDONLY);

    // TODO: We assume 32 pages is enough for now
    const size_t max_size = 4096*32;
    char *buf = (char *)memalign(4096, max_size);
    size_t size = read(fd, buf, max_size);
    if (size == max_size) abort();
    close(fd);

    elf_t *elf = (elf_t *)malloc(sizeof(elf_t));
    elf->parse_object_file(buf, size);
    elf->do_relocations();
    //elf->dump_as_base64();
    //elf->dump_to_file("/tmp/relocated.o");
    //elf->make_executable();
    return (void *)elf;
}

inline void *fake_dlsym(void *handle, const char *name) {
    printf("fake dlsym lookup of %s\n", name);
    elf_t *elf = (elf_t *)handle;
    if (!elf) abort();
    symbol_t *sym = elf->find_symbol(name);
    if (!sym) return NULL;
    if (!elf->symbol_is_defined(sym)) return NULL;
    return (void *)elf->get_symbol_addr(sym);
}

inline int fake_dlclose(void *handle) {
    elf_t *elf = (elf_t *)handle;
    free(elf->buf);
    free(elf);
    return 0;
}