#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 100

extern int test_dict_operations();

typedef struct Entry {
    char* key;
    int value;
    struct Entry* next;
} Entry;

typedef struct HashMap {
    Entry* buckets[TABLE_SIZE];
} HashMap;

unsigned int hash(const char* key) {
    unsigned long hash = 5381;
    int c;
    while ((c = *key++))
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    return hash % TABLE_SIZE;
}

HashMap* create_map() {
    HashMap* map = malloc(sizeof(HashMap));
    for (int i = 0; i < TABLE_SIZE; ++i)
        map->buckets[i] = NULL;
    return map;
}

void put(HashMap* map, const char* key, int value) {
    unsigned int index = hash(key);
    Entry* entry = map->buckets[index];

    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            entry->value = value;
            return;
        }
        entry = entry->next;
    }

    Entry* new_entry = malloc(sizeof(Entry));
    new_entry->key = strdup(key);
    new_entry->value = value;
    new_entry->next = map->buckets[index];
    map->buckets[index] = new_entry;
}

int* get(HashMap* map, const char* key) {
    unsigned int index = hash(key);
    Entry* entry = map->buckets[index];
    while (entry) {
        if (strcmp(entry->key, key) == 0)
            return &entry->value;
        entry = entry->next;
    }
    return NULL;
}

void delete(HashMap* map, const char* key) {
    unsigned int index = hash(key);
    Entry* entry = map->buckets[index];
    Entry* prev = NULL;
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            if (prev)
                prev->next = entry->next;
            else
                map->buckets[index] = entry->next;
            free(entry->key);
            free(entry);
            return;
        }
        prev = entry;
        entry = entry->next;
    }
}

void free_map(HashMap* map) {
    for (int i = 0; i < TABLE_SIZE; ++i) {
        Entry* entry = map->buckets[i];
        while (entry) {
            Entry* temp = entry;
            entry = entry->next;
            free(temp->key);
            free(temp);
        }
    }
    free(map);
}

int main() {
    printf("Starting dictionary test...\n");
    int result = test_dict_operations();
    printf("Result from test_dict_operations: %d\n", result);
    
    return 0;
}
