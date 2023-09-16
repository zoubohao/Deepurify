import gzip
import os
import sys
import stat
import subprocess
import logging
import shutil

import numpy as np


def readFasta(fastaFile, trimHeader=True):
    '''Read sequences from FASTA file.'''
    try:
        openFile = gzip.open if fastaFile.endswith('.gz') else open
        seqs = {}
        for line in openFile(fastaFile, 'rt'):
            # skip blank lines
            if not line.strip():
                continue

            if line[0] == '>':
                seqId = line[1:].split(None, 1)[0] if trimHeader else line[1:].rstrip()
                seqs[seqId] = []
            else:
                seqs[seqId].append(line[:-1])

        for seqId, seq in seqs.items():
            seqs[seqId] = ''.join(seq)
    except Exception as e:
        print(e)
        logger = logging.getLogger('timestamp')
        logger.error(f"Failed to process sequence file: {fastaFile}")
        sys.exit(1)

    return seqs


def checkFileExists(inputFile):
    """Check if file exists."""
    if not os.path.exists(inputFile):
        logger = logging.getLogger('timestamp')
        logger.error(f'Input file does not exists: {inputFile}' + '\n')
        sys.exit(1)
        

class ProdigalRunner():
    """Wrapper for running prodigal."""

    def __init__(self, bin_profix, outDir):
        self.logger = logging.getLogger('timestamp')

        # make sure prodigal is installed
        self.checkForProdigal()

        self.aaGeneFile = os.path.join(outDir, f'{bin_profix}.faa')
        self.gffFile = os.path.join(outDir, f'{bin_profix}.gff')

    def run(self, query):

        prodigal_input = query

        # gather statistics about query file
        seqs = readFasta(prodigal_input)
        totalBases = sum(len(seq) for seqId, seq in seqs.items())
        # call ORFs with different translation tables and select the one with the highest coding density
        tableCodingDensity = {}
        for translationTable in [4, 11]:
            aaGeneFile = f'{self.aaGeneFile}.{str(translationTable)}'
            gffFile = f'{self.gffFile}.{str(translationTable)}'

            # check if there is sufficient bases to calculate prodigal parameters
            procedureStr = 'meta' if totalBases < 100000 else 'single'
            cmd = ('prodigal -p %s -q -m -f gff -g %d -a %s -i %s > %s 2> /dev/null' % (procedureStr,
                                                                                        translationTable,
                                                                                        aaGeneFile,
                                                                                        prodigal_input,
                                                                                        gffFile))

            os.system(cmd)

            if not self._areORFsCalled(aaGeneFile) and procedureStr == 'single':
                # prodigal will fail to learn a model if the input genome has a large number of N's
                # so try gene prediction with 'meta'
                cmd = cmd.replace('-p single', '-p meta')
                os.system(cmd)

            # determine coding density
            prodigalParser = ProdigalGeneFeatureParser(gffFile)

            codingBases = sum(
                prodigalParser.codingBases(seqId) for seqId, seq in seqs.items()
            )
            codingDensity = float(codingBases) / totalBases if totalBases != 0 else 0
            tableCodingDensity[translationTable] = codingDensity

        # determine best translation table
        bestTranslationTable = 11
        if (tableCodingDensity[4] - tableCodingDensity[11] > 0.05) and tableCodingDensity[4] > 0.7:
            bestTranslationTable = 4

        shutil.copyfile(f'{self.aaGeneFile}.{bestTranslationTable}', self.aaGeneFile)
        shutil.copyfile(f'{self.gffFile}.{bestTranslationTable}', self.gffFile)

        # clean up redundant prodigal results
        for translationTable in [4, 11]:
            os.remove(f'{self.aaGeneFile}.{str(translationTable)}')
            os.remove(f'{self.gffFile}.{str(translationTable)}')

        return bestTranslationTable

    def _areORFsCalled(self, aaGeneFile):
        return os.path.exists(aaGeneFile) and os.stat(aaGeneFile)[stat.ST_SIZE] != 0

    def checkForProdigal(self):
        """Check to see if Prodigal is on the system before we try to run it."""

        # Assume that a successful prodigal -h returns 0 and anything
        # else returns something non-zero
        try:
            subprocess.call(
                ['prodigal', '-h'], stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)
        except:
            self.logger.error("Make sure prodigal is on your system path.")
            sys.exit(1)


class ProdigalFastaParser():
    """Parses prodigal FASTA output."""

    def __init__(self):
        pass

    def genePositions(self, filename):
        checkFileExists(filename)

        gp = {}
        for line in open(filename):
            if line[0] == '>':
                lineSplit = line[1:].split()

                geneId = lineSplit[0]
                startPos = int(lineSplit[2])
                endPos = int(lineSplit[4])

                gp[geneId] = [startPos, endPos]

        return gp


class ProdigalGeneFeatureParser():
    """Parses prodigal FASTA output."""

    def __init__(self, filename):
        checkFileExists(filename)

        self.genes = {}
        self.lastCodingBase = {}

        self._parseGFF(filename)

        self.codingBaseMasks = {}
        for seqId in self.genes:
            self.codingBaseMasks[seqId] = self._buildCodingBaseMask(seqId)

    def _parseGFF(self, filename):
        """Parse genes from GFF file."""
        self.translationTable = None
        for line in open(filename):
            if line.startswith('# Model Data') and not self.translationTable:
                lineSplit = line.split(';')
                for token in lineSplit:
                    if 'transl_table' in token:
                        self.translationTable = int(
                            token[token.find('=') + 1:])

            if line[0] == '#' or line.strip() == '"':
                # work around for Prodigal having lines with just a
                # quotation on it when FASTA files have Windows style
                # line endings
                continue

            lineSplit = line.split('\t')
            seqId = lineSplit[0]
            if seqId not in self.genes:
                geneCounter = 0
                self.genes[seqId] = {}
                self.lastCodingBase[seqId] = 0

            geneId = f'{seqId}_{str(geneCounter)}'
            geneCounter += 1

            start = int(lineSplit[3])
            end = int(lineSplit[4])

            self.genes[seqId][geneId] = [start, end]
            self.lastCodingBase[seqId] = max(self.lastCodingBase[seqId], end)

    def _buildCodingBaseMask(self, seqId):
        """Build mask indicating which bases in a sequences are coding."""

        # safe way to calculate coding bases as it accounts
        # for the potential of overlapping genes; indices adjusted
        # to account for GFF file using 1-based indexing
        codingBaseMask = np.zeros(self.lastCodingBase[seqId])
        for pos in self.genes[seqId].values():
            codingBaseMask[pos[0]-1:pos[1]] = 1

        return codingBaseMask

    def codingBases(self, seqId, start=0, end=None):
        """Calculate number of coding bases in sequence between [start, end)."""

        # check if sequence has any genes
        if seqId not in self.genes:
            return 0

        # set end to last coding base if not specified
        if end is None:
            end = self.lastCodingBase[seqId]

        return np.sum(self.codingBaseMasks[seqId][start:end])