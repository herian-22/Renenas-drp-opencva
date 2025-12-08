/***********************************************************************************************************************
* File Name    : dmabuf.cpp
* Version      : v3.10 (Zero-Copy Optimized)
* Description  : DMA Buffer Manager without CPU Overhead
***********************************************************************************************************************/

#include "dmabuf.h"
// #include <cstring> // Hapus cstring, kita tidak pakai memset lagi (CPU Killer)

/*****************************************
* Function Name : buffer_alloc_dmabuf
* Description   : Allocate a DMA buffer in continuous memory area.
* OPTIMIZATION: REMOVED MEMSET TO PREVENT CPU BOTTLENECK
******************************************/
int8_t buffer_alloc_dmabuf(dma_buffer *buffer, int buf_size)
{
    MMNGR_ID id;
    uint32_t phard_addr;
    void *puser_virt_addr;
    int m_dma_fd;

    buffer->size = buf_size;

    // Alokasi memori CMA (Continuous Memory Area)
    // MMNGR_VA_SUPPORT_CACHED tetap digunakan karena nanti OpenCV (CPU) 
    // masih perlu akses untuk Remap. Jika Uncached, akses CPU akan sangat lambat.
    int ret = mmngr_alloc_in_user_ext(&id, buffer->size, &phard_addr, &puser_virt_addr, MMNGR_VA_SUPPORT_CACHED, NULL);
    
    if (ret != 0) {
        return -1;
    }

    // --- OPTIMASI NOL COPY ---
    // HAPUS BARIS INI: memset((void*)puser_virt_addr, 0, buffer->size);
    // Penjelasan: 
    // Mengisi 0 pada buffer 2MP (FHD) memakan waktu CPU yang signifikan (20-40ms di embedded).
    // Karena buffer ini akan langsung diisi data kamera, inisialisasi 0 tidak berguna.
    // -------------------------

    buffer->idx = id;
    buffer->mem = (void *)puser_virt_addr;
    buffer->phy_addr = phard_addr;
    
    if (!buffer->mem)
    {
        return -1;
    }

    // Dapatkan File Descriptor (FD) untuk DRP-AI
    mmngr_export_start_in_user_ext(&id, buffer->size, phard_addr, &m_dma_fd, NULL);
    buffer->dbuf_fd = m_dma_fd; // Pastikan struct dma_buffer di header punya field ini, atau gunakan logic mapping lain
    
    return 0;
}

/*****************************************
* Function Name : buffer_free_dmabuf
******************************************/
void buffer_free_dmabuf(dma_buffer *buffer)
{
    mmngr_free_in_user_ext(buffer->idx);
    return;
}

/*****************************************
* Function Name : buffer_flush_dmabuf
* Description   : Flush cache agar data di RAM terlihat oleh Hardware (DRP-AI)
******************************************/
int buffer_flush_dmabuf(uint32_t idx, uint32_t size)
{
    // Flush wajib dilakukan karena kita menggunakan MMNGR_VA_SUPPORT_CACHED.
    // Ini memberitahu CPU untuk menulis data dari L1/L2 Cache ke RAM fisik
    // agar DRP-AI bisa membacanya.
    return mmngr_flush(idx, 0, size);
}